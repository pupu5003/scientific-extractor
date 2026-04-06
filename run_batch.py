"""
run_batch.py
Batch runner: processes multiple PDFs concurrently in a single Python process.

Usage:
    python3 run_batch.py <glob_or_dir> [options]

Examples:
    python3 run_batch.py "tests/pdfs/iclr2025/spotlight/*.pdf" --llm_backend together
    python3 run_batch.py tests/pdfs/iclr2025/spotlight/ --pdf_workers 4
"""
import asyncio
import argparse
import glob
import json
import os
import shlex
import sys
import tempfile
import time
from pathlib import Path
from dotenv import load_dotenv
from src.extract_references.clients import AsyncLLMClient
from src.extract_references.pipeline import ExtractionPipeline


def resolve_pdfs(path_pattern: str) -> list[str]:
    """Accepts a glob pattern or a directory, returns sorted list of PDF paths."""
    if os.path.isdir(path_pattern):
        pdfs = glob.glob(os.path.join(path_pattern, "*.pdf"))
    else:
        pdfs = glob.glob(path_pattern)
    pdfs = sorted(p for p in pdfs if p.endswith(".pdf"))
    return pdfs


async def process_one_markdown(
    pdf_path: str,
    markdown_path: Path,
    pipeline: ExtractionPipeline,
    output_dir: str,
    sem: asyncio.Semaphore,
    idx: int,
    total: int,
) -> bool:
    """Process a single markdown file under a concurrency semaphore, save output."""
    async with sem:
        base_name = os.path.basename(pdf_path)
        print(f"[{idx}/{total}] Parsing Markdown: {base_name}")
        t0 = time.monotonic()
        try:
            markdown_text = markdown_path.read_text(encoding="utf-8", errors="ignore")
            results = await pipeline.run_from_markdown(markdown_text, source_name=base_name)
            output_data = [r.model_dump(exclude_none=True) for r in results]
            os.makedirs(output_dir, exist_ok=True)
            out_file = os.path.join(output_dir, f"{base_name}_extracted.json")
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            elapsed = time.monotonic() - t0
            print(f"[{idx}/{total}] Done: {base_name} → {len(results)} refs ({elapsed:.1f}s)")
            return True
        except Exception as e:
            elapsed = time.monotonic() - t0
            print(f"[{idx}/{total}] FAILED: {base_name} — {e} ({elapsed:.1f}s)", file=sys.stderr)
            return False


def _build_markdown_path_map(mineru_out_dir: Path, pdfs: list[str], method: str = "txt") -> dict[str, Path]:
    """Map each PDF path to its MinerU markdown output path."""
    result: dict[str, Path] = {}
    for pdf in pdfs:
        stem = Path(pdf).stem
        md_path = mineru_out_dir / stem / method / f"{stem}.md"
        if md_path.exists():
            result[pdf] = md_path
    return result


async def _run_mineru_once_for_folder(args, pdf_root: str) -> Path:
    """Run MinerU one-shot on a directory of PDFs and return output root."""
    if args.mineru_batch_output_dir:
        out_dir = Path(args.mineru_batch_output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path(tempfile.mkdtemp(prefix="mineru_batch_"))

    quoted_pdf_root = shlex.quote(pdf_root)
    quoted_out = shlex.quote(str(out_dir))
    cmd = args.mineru_cmd.format(pdf=quoted_pdf_root, out_dir=quoted_out)
    print(f"[Batch] Running MinerU once on folder: {pdf_root}")
    print(f"[Batch] Command: {cmd}")

    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"MinerU batch failed (exit {proc.returncode})\n"
            f"STDERR:\n{stderr.decode('utf-8', errors='ignore')}\n"
            f"STDOUT:\n{stdout.decode('utf-8', errors='ignore')}"
        )

    return out_dir


async def main_async(args):
    pdfs = resolve_pdfs(args.input)
    if not pdfs:
        print(f"No PDF files found at: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(pdfs)} PDFs. Running one MinerU folder pass, then markdown parsing with {args.pdf_workers} workers...\n")

    # Build LLM client
    if args.llm_backend == "together":
        api_key = os.environ.get("TOGETHER_API_KEY", "")
        base_url = "https://api.together.xyz/v1"
        model = args.model if args.model != "gpt-4o-mini" else "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
    elif args.llm_backend == "ollama":
        api_key = "ollama_placeholder"
        base_url = "http://localhost:11434/v1"
        model = args.model
    else:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        base_url = None
        model = args.model

    llm_client = AsyncLLMClient(api_key=api_key, base_url=base_url, model=model)

    # Single shared pipeline — citation parsing is concurrent
    pipeline = ExtractionPipeline(
        mineru_cmd=args.mineru_cmd,
        llm_client=llm_client,
        max_concurrency=args.citation_workers,
        debug_markdown_dir=args.debug_markdown_dir or None,
    )

    # Run MinerU once for the whole folder (or parent folder for glob input)
    if os.path.isdir(args.input):
        pdf_root = args.input
    else:
        pdf_root = os.path.dirname(os.path.abspath(pdfs[0])) or "."

    mineru_out_dir = await _run_mineru_once_for_folder(args, pdf_root)
    md_map = _build_markdown_path_map(mineru_out_dir, pdfs, method=args.mineru_method)

    missing = [p for p in pdfs if p not in md_map]
    if missing:
        print(f"[Batch] Warning: Missing markdown outputs for {len(missing)} PDFs")
        for m in missing[:10]:
            print(f"  - {m}")

    tasks_input = [(pdf, md_map[pdf]) for pdf in pdfs if pdf in md_map]

    sem = asyncio.Semaphore(args.pdf_workers)
    t_start = time.monotonic()
    tasks = [
        process_one_markdown(pdf, md_path, pipeline, args.output_dir, sem, i + 1, len(tasks_input))
        for i, (pdf, md_path) in enumerate(tasks_input)
    ]
    outcomes = await asyncio.gather(*tasks)

    total_time = time.monotonic() - t_start
    ok = sum(outcomes)
    failed = len(outcomes) - ok
    print(f"\n{'='*50}")
    print(f"Completed: {ok}/{len(tasks_input)} parsed Markdown files in {total_time:.1f}s")
    if failed:
        print(f"Failed: {failed} PDFs — check stderr above.")
    print(f"Output saved to: {args.output_dir}")


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Batch PDF reference extractor")
    parser.add_argument("input", help="Glob pattern or directory of PDFs, e.g. 'tests/pdfs/*.pdf'")
    parser.add_argument("--output_dir", default="tests/json/batch_output/", help="Output directory for JSON files")
    parser.add_argument(
        "--mineru_batch_output_dir",
        default="",
        help="If set, MinerU one-shot output will be stored here instead of a temp folder.",
    )
    parser.add_argument(
        "--mineru_cmd",
        default="mineru -p {pdf} -o {out_dir}",
        help="MinerU command template. Must contain {pdf} and {out_dir} placeholders.",
    )
    parser.add_argument("--llm_backend", default="openai", choices=["openai", "ollama", "together"])
    parser.add_argument(
        "--debug_markdown_dir",
        default="",
        help="If set, save MinerU markdown output (.md) for each input PDF into this directory.",
    )
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument(
        "--mineru_method",
        default="txt",
        choices=["txt", "auto", "ocr"],
        help="Expected MinerU method subfolder name to read markdown from (default: txt).",
    )
    parser.add_argument(
        "--pdf_workers", type=int, default=3,
        help="Max PDFs processed concurrently (default: 3). Higher = faster but more load on MinerU/anystyle."
    )
    parser.add_argument(
        "--citation_workers", type=int, default=10,
        help="Max citations parsed concurrently per PDF (default: 10)."
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
