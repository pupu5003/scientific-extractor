#!/usr/bin/env python3
"""
extract_refs_from_content_list.py
----------------------------------
Standalone script: extract the References section from a MinerU
content_list.json file and write the results to disk.

Usage
-----
  python scripts/extract_refs_from_content_list.py \\
      tests/mineru_output/2512.24784v1/txt/2512.24784v1_content_list.json

Options
-------
  --output-json   Path for the output JSON list  (default: <stem>_refs.json)
  --output-txt    Path for plain-text output      (default: <stem>_refs.txt)
  --no-txt        Skip writing the .txt file
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from extract_references.clients import AsyncMinerUClient


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract References section from a MinerU content_list.json file."
    )
    parser.add_argument(
        "content_list",
        help="Path to the *_content_list.json produced by MinerU.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Output path for the JSON list of references (default: <stem>_refs.json).",
    )
    parser.add_argument(
        "--output-txt",
        default=None,
        help="Output path for the plain-text list of references (default: <stem>_refs.txt).",
    )
    parser.add_argument(
        "--no-txt",
        action="store_true",
        help="Skip writing the plain-text output file.",
    )
    args = parser.parse_args()

    content_list_path = Path(args.content_list).resolve()
    if not content_list_path.exists():
        print(f"[ERROR] File not found: {content_list_path}", file=sys.stderr)
        sys.exit(1)

    # Default output paths sit next to the input file
    stem = content_list_path.stem.replace("_content_list", "")
    out_dir = content_list_path.parent

    json_out = Path(args.output_json) if args.output_json else out_dir / f"{stem}_refs.json"
    txt_out  = Path(args.output_txt)  if args.output_txt  else out_dir / f"{stem}_refs.txt"

    # ── Load & Extract ──────────────────────────────────────────────────
    print(f"[extract] Loading {content_list_path.name} …")
    content_list: list[dict] = json.loads(
        content_list_path.read_text(encoding="utf-8")
    )

    client = AsyncMinerUClient()
    refs = client.extract_references_from_content_list(content_list)

    if not refs:
        print("[extract] No references found. Check that the PDF has a 'References' section.")
        sys.exit(0)

    print(f"[extract] Found {len(refs)} reference(s).")

    # ── Write JSON ───────────────────────────────────────────────────────
    json_out.parent.mkdir(parents=True, exist_ok=True)
    # Each entry is {"ref_id": "R1", "raw_text": "..."}
    payload = [
        {"ref_id": f"R{i}", "raw_text": ref}
        for i, ref in enumerate(refs, 1)
    ]
    json_out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[extract] JSON  -> {json_out}")

    # ── Write TXT ────────────────────────────────────────────────────────
    if not args.no_txt:
        txt_out.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"[R{i}] {ref}" for i, ref in enumerate(refs, 1)]
        txt_out.write_text("\n\n".join(lines), encoding="utf-8")
        print(f"[extract] TXT   -> {txt_out}")


if __name__ == "__main__":
    main()
