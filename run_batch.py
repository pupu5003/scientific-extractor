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


async def process_one_pdf(
    pdf_path: str,
    pipeline: ExtractionPipeline,
    output_dir: str,
    sem: asyncio.Semaphore,
    idx: int,
    total: int,
) -> bool:
    """Process a single PDF under a concurrency semaphore, save output."""
    async with sem:
        base_name = os.path.basename(pdf_path)
        print(f"[{idx}/{total}] Starting PDF: {base_name}")
        t0 = time.monotonic()
        try:
            result = await pipeline.run(pdf_path)
            output_data = result.model_dump(exclude_none=True)
            os.makedirs(output_dir, exist_ok=True)

            # Use same stem logic as __main__.py
            stem = Path(pdf_path).stem
            out_file = os.path.join(output_dir, f"{stem}_extracted.json")

            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)

            elapsed = time.monotonic() - t0
            print(
                f"[{idx}/{total}] Done: {base_name} → {len(result.references)} refs, "
                f"{len(result.claims)} claims ({elapsed:.1f}s)"
            )
            return True
        except Exception as e:
            elapsed = time.monotonic() - t0
            print(f"[{idx}/{total}] FAILED: {base_name} — {e} ({elapsed:.1f}s)", file=sys.stderr)
            return False


async def main_async(args):
    pdfs = resolve_pdfs(args.input)
    if not pdfs:
        print(f"No PDF files found at: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(pdfs)} PDFs. Processing with {args.pdf_workers} workers (Parallel PDFs)...\n")

    # Build LLM client
    if args.llm_backend == "together":
        api_key = os.environ.get("TOGETHER_API_KEY", "")
        base_url = "https://api.together.xyz/v1"
        model = args.model
        use_json_mode = True
    elif args.llm_backend == "ollama":
        api_key = "ollama_placeholder"
        base_url = "http://localhost:11434/v1"
        model = args.model
        use_json_mode = True
    else:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        base_url = None
        model = args.model
        use_json_mode = False

    llm_client = AsyncLLMClient(api_key=api_key, base_url=base_url, model=model, use_json_mode=use_json_mode)

    # Single shared pipeline — citation parsing is also concurrent within each PDF
    pipeline = ExtractionPipeline(
        mineru_cmd=args.mineru_cmd,
        llm_client=llm_client,
        max_concurrency=args.citation_workers,
        debug_markdown_dir=args.debug_markdown_dir or None,
        skip_claims=args.no_claims,
    )

    sem = asyncio.Semaphore(args.pdf_workers)
    t_start = time.monotonic()
    
    tasks = [
        process_one_pdf(pdf, pipeline, args.output_dir, sem, i + 1, len(pdfs))
        for i, pdf in enumerate(pdfs)
    ]
    outcomes = await asyncio.gather(*tasks)

    total_time = time.monotonic() - t_start
    ok = sum(outcomes)
    failed = len(outcomes) - ok
    print(f"\n{'='*50}")
    print(f"Completed: {ok}/{len(pdfs)} PDFs in {total_time:.1f}s")
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
        default=".venv/bin/mineru -p {pdf} -o {out_dir} -b pipeline -m txt -d cpu -f false -t false",
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
        "--pdf_workers", type=int, default=5,
        help="Max PDFs processed concurrently (default: 5). Higher = faster but more load on MinerU/CPU."
    )
    parser.add_argument(
        "--citation_workers", type=int, default=10,
        help="Max citations parsed concurrently per PDF, and max claim-extraction "
             "paragraphs parsed concurrently per PDF (default: 10)."
    )
    parser.add_argument(
        "--no_claims",
        action="store_true",
        help="Skip LLM-based claim extraction from the body text — references only.",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
