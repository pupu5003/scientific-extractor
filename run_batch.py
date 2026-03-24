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
import sys
import time
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
        print(f"[{idx}/{total}] Starting: {base_name}")
        t0 = time.monotonic()
        try:
            results = await pipeline.run(pdf_path)
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


async def main_async(args):
    pdfs = resolve_pdfs(args.input)
    if not pdfs:
        print(f"No PDF files found at: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(pdfs)} PDFs. Running with {args.pdf_workers} concurrent PDFs...\n")

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

    # Single shared pipeline — citations within each PDF are already concurrent
    pipeline = ExtractionPipeline(
        grobid_url=args.grobid_url,
        llm_client=llm_client,
        max_concurrency=args.citation_workers,
    )

    # Semaphore limits how many PDFs are processed at the same time
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
    parser.add_argument("--grobid_url", default="http://localhost:8070")
    parser.add_argument("--llm_backend", default="openai", choices=["openai", "ollama", "together"])
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument(
        "--pdf_workers", type=int, default=3,
        help="Max PDFs processed concurrently (default: 3). Higher = faster but more load on GROBID/anystyle."
    )
    parser.add_argument(
        "--citation_workers", type=int, default=10,
        help="Max citations parsed concurrently per PDF (default: 10)."
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
