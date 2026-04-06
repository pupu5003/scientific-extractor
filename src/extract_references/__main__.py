"""
__main__.py
CLI entry point for the extraction module.

Usage:
  # New flow — pass the MinerU output folder (auto-finds content_list.json)
  python3 -m src.extract_references tests/mineru_output/2512.24784v1/ --llm_backend together

  # Legacy flow — pass a PDF directly (mineru in .venv)
  python3 -m src.extract_references paper.pdf --llm_backend openai
"""
from __future__ import annotations

import asyncio
import argparse
import sys
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from .clients import AsyncLLMClient
from .pipeline import ExtractionPipeline

DEFAULT_MINERU_CMD = (
    ".venv/bin/mineru -p {pdf} -o {out_dir} -b pipeline -m txt -d cpu -f false -t false"
)


def _find_content_list(folder: Path) -> Path | None:
    """Search for *_content_list.json inside a MinerU output folder."""
    for candidate in [folder / "txt", folder]:
        matches = sorted(candidate.glob("*_content_list.json"))
        if matches:
            return matches[0]
    return None


def _build_llm_client(args: argparse.Namespace) -> AsyncLLMClient:
    if args.llm_backend == "together":
        api_key = os.environ.get("TOGETHER_API_KEY", "")
        base_url = "https://api.together.xyz/v1"
        model = args.model if args.model != "gpt-4o-mini" else "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
    elif args.llm_backend == "ollama":
        api_key = "ollama_placeholder"
        base_url = "http://localhost:11434/v1"
        model = args.model
    else:  # openai
        api_key = os.environ.get("OPENAI_API_KEY", "")
        base_url = None
        model = args.model
    return AsyncLLMClient(api_key=api_key, base_url=base_url, model=model)


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Scientific Reference Extractor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Pass MinerU output folder (recommended)
  python3 -m src.extract_references tests/mineru_output/2512.24784v1/ --llm_backend openai

  # Pass a PDF directly (runs mineru via .venv)
  python3 -m src.extract_references paper.pdf --llm_backend openai

Default mineru command:
  {DEFAULT_MINERU_CMD}
""",
    )

    parser.add_argument(
        "input_path",
        help=(
            "MinerU output folder (auto-finds *_content_list.json inside) "
            "OR a PDF file path (runs MinerU via --mineru_cmd)."
        ),
    )
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
        help="[PDF mode only] MinerU command template. Must contain {pdf} and {out_dir}.",
    )

    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Error: path not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    llm_client = _build_llm_client(args)
    pipeline = ExtractionPipeline(
        mineru_cmd=args.mineru_cmd,
        llm_client=llm_client,
        max_concurrency=args.concurrency,
    )

    try:
        if input_path.is_dir():
            # ── NEW FLOW: folder → auto-find content_list.json ─────────
            content_list_path = _find_content_list(input_path)
            if not content_list_path:
                print(
                    f"Error: no *_content_list.json found in {input_path} (checked txt/ subfolder too).",
                    file=sys.stderr,
                )
                sys.exit(1)
            print(f"[Pipeline] Using: {content_list_path}")
            results = asyncio.run(
                pipeline.run_from_content_list(str(content_list_path))
            )
            stem = content_list_path.stem.replace("_content_list", "")

        else:
            # ── LEGACY FLOW: PDF → run MinerU ──────────────────────────
            results = asyncio.run(pipeline.run(str(input_path)))
            stem = input_path.stem

        # ── Write final extracted JSON ──────────────────────────────────
        output_data = [res.model_dump(exclude_none=True) for res in results]
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir, f"{stem}_extracted.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"\n[Success] Processed {len(results)} citations.")
        print(f"[Success] Saved → {output_file}")

    except Exception as e:
        print(f"Pipeline execution failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()