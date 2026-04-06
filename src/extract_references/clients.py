"""
clients.py
Async clients for MinerU extraction, anystyle parsing, and LLM providers.
"""
from __future__ import annotations
import asyncio
import json
import os
import re
import shlex
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential
import instructor
from openai import AsyncOpenAI
from .schemas import LLMPatchInstruction, ParsedCitationEntry, CitationCollection


class AsyncMinerUClient:
    def __init__(self, command_template: Optional[str] = None, debug_markdown_dir: Optional[str] = None):
        self.command_template = command_template or "mineru -p {pdf} -o {out_dir}"
        self.debug_markdown_dir = debug_markdown_dir

    @staticmethod
    def _command_exists(command_template: str) -> bool:
        try:
            first_token = shlex.split(command_template)[0]
        except Exception:
            return False
        return shutil.which(first_token) is not None

    @staticmethod
    def _ensure_cli_on_path(command_template: str) -> None:
        """Ensure venv bin is on PATH so `mineru` installed there is discoverable."""
        try:
            first_token = shlex.split(command_template)[0]
        except Exception:
            return
        if shutil.which(first_token) is not None:
            return
        venv_bin = Path(sys.executable).resolve().parent
        current_path = os.environ.get("PATH", "")
        if str(venv_bin) not in current_path:
            os.environ["PATH"] = f"{venv_bin}{os.pathsep}{current_path}"

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=5), reraise=True)
    async def extract_raw_references(self, pdf_path: str) -> List[str]:
        """Run MinerU on *pdf_path* and extract references via content_list.json."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        self._ensure_cli_on_path(self.command_template)
        if not self._command_exists(self.command_template):
            raise RuntimeError(
                "MinerU CLI not found in PATH. Ensure your venv is active or pass a valid command via --mineru_cmd. "
                "Tried command template: "
                f"{self.command_template}"
            )

        with tempfile.TemporaryDirectory(prefix="mineru_out_") as out_dir:
            quoted_pdf = shlex.quote(pdf_path)
            quoted_out_dir = shlex.quote(out_dir)
            cmd = self.command_template.format(pdf=quoted_pdf, out_dir=quoted_out_dir)
            print(f"[MinerU] Running command: {cmd}")
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=180)
            if proc.returncode != 0:
                err = stderr.decode("utf-8", errors="ignore")
                raise RuntimeError(f"MinerU failed (exit {proc.returncode}): {err}")

            # Prefer content_list.json — more accurate than markdown
            cl_files = sorted(Path(out_dir).rglob("*_content_list.json"))
            if cl_files:
                best_cl = max(cl_files, key=lambda p: p.stat().st_size)
                content_list = json.loads(best_cl.read_text(encoding="utf-8"))
                return self.extract_references_from_content_list(content_list)

            # Fallback: markdown (should be rare)
            md_files = sorted(Path(out_dir).rglob("*.md"))
            if not md_files:
                raise RuntimeError("MinerU completed but no output was found (neither content_list.json nor *.md)")
            best_md = max(md_files, key=lambda p: p.stat().st_size)
            markdown_text = best_md.read_text(encoding="utf-8", errors="ignore")
            return self.extract_references_from_markdown(markdown_text)

    async def extract_markdown(self, pdf_path: str) -> str:
        """Runs MinerU on PDF and returns produced markdown content."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        self._ensure_cli_on_path(self.command_template)
        if not self._command_exists(self.command_template):
            raise RuntimeError(
                "MinerU CLI not found in PATH. Ensure your venv is active or pass a valid command via --mineru_cmd. "
                "Tried command template: "
                f"{self.command_template}"
            )

        with tempfile.TemporaryDirectory(prefix="mineru_out_") as out_dir:
            quoted_pdf = shlex.quote(pdf_path)
            quoted_out_dir = shlex.quote(out_dir)
            cmd = self.command_template.format(pdf=quoted_pdf, out_dir=quoted_out_dir)
            print(f"[MinerU] Running command: {cmd}")
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=180)
            if proc.returncode != 0:
                err = stderr.decode("utf-8", errors="ignore")
                raise RuntimeError(f"MinerU failed (exit {proc.returncode}): {err}")

            md_files = sorted(Path(out_dir).rglob("*.md"))
            if not md_files:
                raise RuntimeError("MinerU completed but no markdown (*.md) output was found")

            best_md = max(md_files, key=lambda p: p.stat().st_size)
            markdown_text = best_md.read_text(encoding="utf-8", errors="ignore")

            if self.debug_markdown_dir:
                os.makedirs(self.debug_markdown_dir, exist_ok=True)
                pdf_name = Path(pdf_path).stem

                # Save full markdown for inspection
                debug_file = Path(self.debug_markdown_dir) / f"{pdf_name}.md"
                debug_file.write_text(markdown_text, encoding="utf-8")

                # Save references-only markdown
                refs_block = self.extract_references_block_markdown(markdown_text)
                refs_file = Path(self.debug_markdown_dir) / f"{pdf_name}_references.md"
                refs_file.write_text(refs_block, encoding="utf-8")

                # Clean everything except markdown outputs in the debug folder
                for extra in Path(self.debug_markdown_dir).glob(f"{pdf_name}*"):
                    if extra.suffix not in {".md"}:
                        try:
                            extra.unlink()
                        except Exception:
                            pass

            return markdown_text

    def extract_references_from_markdown(self, markdown_text: str) -> List[str]:
        """Extract references section from markdown and split into entries."""
        lines = markdown_text.splitlines()

        start_idx = self._find_references_start(lines)
        if start_idx is None:
            return []

        ref_lines = self._collect_references_block(lines, start_idx)
        if not ref_lines:
            return []

        # Join and use the smart splitter
        block = "\n".join(ref_lines)
        parsed = self._split_content_list_refs(block)

        cleaned = [self._normalize_reference_text(x) for x in parsed]
        cleaned = [x for x in cleaned if len(x) >= 20]
        return cleaned

    def _find_references_start(self, lines: List[str]) -> Optional[int]:
        headings = {
            "references",
            "reference",
            "bibliography",
            "works cited",
            "citations",
            "tai lieu tham khao",
            "tài liệu tham khảo",
        }
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            normalized = stripped.lower().strip("#").strip()
            normalized = re.sub(r"\s+", " ", normalized)
            if normalized in headings:
                return i + 1
            if re.match(r"^#{1,6}\s*(references?|bibliography|works\s+cited)\s*$", stripped, flags=re.IGNORECASE):
                return i + 1
        return None

    def extract_references_block_markdown(self, markdown_text: str) -> str:
        """Return references section as markdown (with header)."""
        lines = markdown_text.splitlines()
        start_idx = self._find_references_start(lines)
        if start_idx is None:
            return "# References\n"

        ref_lines = self._collect_references_block(lines, start_idx)
        block = "\n".join(ref_lines).strip()
        return "# References\n\n" + block + "\n"

    def _collect_references_block(self, lines: List[str], start_idx: int) -> List[str]:
        ref_lines: List[str] = []
        for line in lines[start_idx:]:
            stripped = line.strip()
            if not stripped:
                ref_lines.append("")
                continue

            # Stop when the next top-level/section heading starts
            if stripped.startswith("#"):
                if re.match(r"^#{1,6}\s*(references?|bibliography|works\s+cited)\s*$", stripped, flags=re.IGNORECASE):
                    continue
                break

            # Hard stop heuristics for obvious post-reference metadata tail
            if re.match(r"^(appendix|acknowledg(e)?ments?|author information|email address)\b", stripped, flags=re.IGNORECASE):
                break

            ref_lines.append(line)
        return ref_lines

    @staticmethod
    def _normalize_reference_text(text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    # ------------------------------------------------------------------
    # content_list.json extraction
    # ------------------------------------------------------------------

    REFERENCE_HEADINGS: frozenset[str] = frozenset({
        "references",
        "reference",
        "bibliography",
        "works cited",
        "citations",
        "tai lieu tham khao",
        "tài liệu tham khảo",
    })

    def extract_references_from_content_list(
        self,
        content_list: list[dict],
    ) -> list[str]:
        """Extract individual reference strings from MinerU's content_list.json."""
        ref_blocks: list[str] = []
        in_references = False

        # Types of blocks to strictly ignore within the references section
        IGNORE_TYPES = {"page_number", "header", "page_footnote", "footer", "discarded"}

        for item in content_list:
            item_type = item.get("type", "text")
            if item_type in IGNORE_TYPES:
                continue

            text: str = item.get("text", "").strip()
            text_level = item.get("text_level")

            if text_level is not None and text:
                heading_norm = re.sub(r"\s+", " ", text.lower()).strip()
                if heading_norm in self.REFERENCE_HEADINGS:
                    in_references = True
                    continue
                elif in_references:
                    # Break on major new sections (Level 1 or 2)
                    if text_level <= 2:
                        break
                continue

            if not in_references:
                continue

            # Collect content from text or list blocks
            if item_type == "text" and text:
                ref_blocks.append(text)
            elif item_type == "list":
                items = item.get("list_items", [])
                for li in items:
                    if li.strip():
                        ref_blocks.append(li.strip())

        if not ref_blocks:
            return []

        # Join with newlines to preserve boundary information for the state machine
        combined = "\n".join(ref_blocks)
        entries = self._split_content_list_refs(combined)

        cleaned = [self._normalize_reference_text(e) for e in entries]
        cleaned = [e for e in cleaned if len(e) >= 15]
        return cleaned

    @classmethod
    def _split_content_list_refs(cls, text: str) -> list[str]:
        """Split a possibly-merged reference block into individual entries."""
        marker_pattern = re.compile(
            r"^\s*(?:\[[^\]]+\]|\(\d+\)|\d+[.)]|•\s+|-\s+)\s*"
        )
        
        # Check if the entire text has any standard delimiters
        has_any_marker = any(marker_pattern.match(line) for line in text.splitlines())

        refs: list[str] = []
        current_buffer: list[str] = []

        def flush():
            if current_buffer:
                combined = " ".join(current_buffer)
                combined = re.sub(r"\s+", " ", combined).strip()
                if combined:
                    refs.append(combined)
                current_buffer.clear()

        lines = text.splitlines()
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                flush()
                continue

            # Case A: Standard Marker detected
            if marker_pattern.match(line):
                flush()
                clean_line = marker_pattern.sub("", line).strip()
                if clean_line:
                    current_buffer.append(clean_line)
                continue

            # Case B: No markers detected in the block -> use Capitals + Period heuristic
            if not has_any_marker:
                if line_stripped and line_stripped[0].isupper():
                    is_likely_new = False
                    if not current_buffer:
                        is_likely_new = True
                    else:
                        prev = current_buffer[-1].strip()
                        # If prev line ends with common "end of citation" markers
                        if re.search(r"(\d{4}[).]?|https?://\S+|doi:\S+|\.)$", prev, re.I):
                            is_likely_new = True
                    
                    if is_likely_new:
                        flush()

            current_buffer.append(line_stripped)

        flush()
        return refs

    @classmethod
    def load_and_extract_references(
        cls,
        content_list_path: str,
        output_path: str | None = None,
    ) -> list[str]:
        import pathlib
        path = pathlib.Path(content_list_path)
        content_list = json.loads(path.read_text(encoding="utf-8"))

        instance = cls()
        refs = instance.extract_references_from_content_list(content_list)

        if output_path:
            out = pathlib.Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("\n\n".join(refs), encoding="utf-8")
            print(f"[MinerUClient] Wrote {len(refs)} references -> {out}")

        return refs


class AsyncLLMClient:
    def __init__(self, api_key: str, base_url: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.client = instructor.from_openai(AsyncOpenAI(api_key=api_key, base_url=base_url))
        self.model = model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def extract_citation(self, raw_text: str) -> ParsedCitationEntry:
        """Use Instructor to extract metadata for a single raw citation string."""
        return await self.client.chat.completions.create(
            model=self.model,
            response_model=ParsedCitationEntry,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a high-precision citation extraction specialist.\n"
                        "Extract metadata from the provided raw text. If information like DOI or arXiv ID is not present, leave it null.\n"
                        "Do not use external knowledge or invent data."
                    )
                },
                {"role": "user", "content": f"raw_text: {raw_text}"},
            ],
        )

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def extract_citations_batch(self, raw_text: str) -> CitationCollection:
        """Asks the LLM to extract multiple references from a block of text at once."""
        return await self.client.chat.completions.create(
            model=self.model,
            response_model=CitationCollection,
            messages=[
                {
                    "role": "system",
                    "content": "Extract each individual bibliographic citation from the text into a structured collection."
                },
                {"role": "user", "content": raw_text},
            ],
        )