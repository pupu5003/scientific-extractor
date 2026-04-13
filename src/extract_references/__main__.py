from __future__ import annotations

import asyncio
import argparse
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from .clients import AsyncLLMClient
from .pipeline import ExtractionPipeline
import json

DEFAULT_MINERU_CMD = (
    ".venv/bin/mineru -p {pdf} -o {out_dir} -b pipeline -m txt -d cpu -f false -t false"
)


def _build_llm_client(args: argparse.Namespace) -> AsyncLLMClient:
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
    else:  # openai
        api_key = os.environ.get("OPENAI_API_KEY", "")
        base_url = None
        model = args.model
        use_json_mode = False
    return AsyncLLMClient(api_key=api_key, base_url=base_url, model=model, use_json_mode=use_json_mode)


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Scientific Reference Extractor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python3 -m src.extract_references paper.pdf --llm_backend openai
  python3 -m src.extract_references paper.pdf --llm_backend together --output_dir out/

Default mineru command:
  {DEFAULT_MINERU_CMD}
""",
    )

    parser.add_argument("pdf_path", help="Path to the PDF file to process.")
    parser.add_argument(
        "--llm_backend",
        type=str,
        default="openai",
        choices=["openai", "ollama", "together"],
    )
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tests/json/minerU/",
        help="Directory for the final *_extracted.json (default: tests/json/minerU/).",
    )
    parser.add_argument(
        "--mineru_cmd",
        type=str,
        default=DEFAULT_MINERU_CMD,
        help="MinerU command template. Must contain {pdf} and {out_dir}.",
    )

    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists() or not pdf_path.is_file():
        print(f"Error: PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    llm_client = _build_llm_client(args)
    pipeline = ExtractionPipeline(
        mineru_cmd=args.mineru_cmd,
        llm_client=llm_client,
        max_concurrency=args.concurrency,
    )

    try:
        results = asyncio.run(pipeline.run(str(pdf_path)))

        output_data = [res.model_dump(exclude_none=True) for res in results]
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir, f"{pdf_path.stem}_extracted.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"\n[Success] Processed {len(results)} citations.")
        print(f"[Success] Saved → {output_file}")

    except Exception as e:
        print(f"Pipeline execution failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()