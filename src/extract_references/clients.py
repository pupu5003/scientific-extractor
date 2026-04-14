"""
clients.py
Async clients for MinerU extraction and LLM providers.
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
from typing import List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
import instructor
from openai import AsyncOpenAI
from .schemas import ParsedCitationEntry, CitationCollection, MergeDecision


def _is_new_ref_start(line: str, current_buffer: list[str]) -> bool:
    """
    Multi-signal heuristic: decide if `line` starts a new reference.
    Requires total score >= 2 to avoid cutting off continuation lines.
    """
    if not line:
        return False
    if not (line[0].isupper() or line[0].isdigit()):
        return False

    prev_text = " ".join(current_buffer)
    signals = 0

    # Signal mạnh (2đ): dòng trước kết thúc bằng year rõ ràng
    if re.search(r"\b(19|20)\d{2}[).,]\s*$", prev_text):
        signals += 2

    # Signal mạnh (2đ): dòng trước kết thúc bằng DOI / URL
    if re.search(r"(doi:\S+|https?://\S+)\s*$", prev_text, re.I):
        signals += 2

    # Signal mạnh (2đ): dòng hiện tại trông như numbered ref không có marker
    if re.match(r"^\d{1,3}\.\s+[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚ]", line):
        signals += 2

    # Signal trung bình (1đ): dòng trước kết thúc page range "pp. 12-34"
    if re.search(r"\b\d+\s*[–—-]\s*\d+\.?\s*$", prev_text):
        signals += 1

    # Signal trung bình (1đ): dòng trước kết thúc journal/volume pattern "12(3)"
    if re.search(r"\b\d+\(\d+\)\.?\s*$", prev_text):
        signals += 1

    # Signal trung bình (1đ): dòng hiện tại bắt đầu như Author surname
    if re.match(
        r"^[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯ][a-zàáâãèéêìíòóôõùúăđĩũơư]+(,\s+|\s+[A-Z]\.\s+|\s+&\s+|\s+and\s+)",
        line,
    ):
        signals += 1

    return signals >= 2


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------

class AsyncMinerUClient:
    """
    Async client wrapping the MinerU CLI.

    Optimize performance:
    - Use command_template to inject flags to turn off formula/table/OCR
    - Example: "mineru -p {pdf} -o {out_dir} -b pipeline -m txt -d cpu -f false -t false"
    - Batch: call extract_raw_references() concurrently with asyncio.gather()
    """

    REFERENCE_HEADINGS: frozenset[str] = frozenset({
        "references",
        "reference",
        "bibliography",
        "works cited",
        "citations",
        "tai lieu tham khao",
        "tài liệu tham khảo",
    })

    # Sections appearing AFTER references → stop collecting
    _POST_REF_STOP = re.compile(
        r"^(appendix|acknowledg|author\s+(information|contribution|note)|"
        r"supplement|conflict\s+of\s+interest|funding|notes?\s*$|annex|"
        r"about\s+the\s+author|ethical\s+approval|declaration)",
        re.IGNORECASE,
    )

    # Types MinerU used for noise — always ignore
    _IGNORE_TYPES: frozenset[str] = frozenset({
        "page_number", "header", "page_footnote", "footer", "discarded",
    })

    # Types MinerU used for headings
    _HEADING_TYPES: frozenset[str] = frozenset({
        "title", "section_header",
    })

    # Types used for reference content
    _CONTENT_TYPES: frozenset[str] = frozenset({
        "text", "list", "list_item", "paragraph",
    })

    def __init__(
        self,
        command_template: Optional[str] = None,
        debug_markdown_dir: Optional[str] = None,
    ):
        self.command_template = (
            command_template
            or "mineru -p {pdf} -o {out_dir} -b pipeline -m txt -d cpu -f false -t false"
        )
        self.debug_markdown_dir = debug_markdown_dir

    # ------------------------------------------------------------------
    # CLI helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _command_exists(command_template: str) -> bool:
        try:
            first_token = shlex.split(command_template)[0]
        except Exception:
            return False
        return shutil.which(first_token) is not None

    @staticmethod
    def _ensure_cli_on_path(command_template: str) -> None:
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

    async def _run_mineru(self, pdf_path: str, out_dir: str) -> None:
        """Run MinerU CLI, raise RuntimeError on failure."""
        quoted_pdf = shlex.quote(pdf_path)
        quoted_out = shlex.quote(out_dir)
        cmd = self.command_template.format(pdf=quoted_pdf, out_dir=quoted_out)
        print(f"[MinerU] Running: {cmd}")
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)
        if proc.returncode != 0:
            err = stderr.decode("utf-8", errors="ignore")
            raise RuntimeError(f"MinerU failed (exit {proc.returncode}): {err}")

    # ------------------------------------------------------------------
    # Public: extract references
    # ------------------------------------------------------------------

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=5), reraise=True)
    async def extract_ref_blocks_with_page_idx(
        self, pdf_path: str
    ) -> list[tuple[str, int]]:
        """
        Run MinerU on *pdf_path* and return list (ref_text, page_idx).
        page_idx is used to detect adjacent pairs that are split across pages.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        self._ensure_cli_on_path(self.command_template)
        if not self._command_exists(self.command_template):
            raise RuntimeError(
                "MinerU CLI not found in PATH. Ensure your venv is active or "
                f"pass a valid command via --mineru_cmd. Template: {self.command_template}"
            )

        with tempfile.TemporaryDirectory(prefix="mineru_out_") as out_dir:
            await self._run_mineru(pdf_path, out_dir)

            # Prioritize content_list.json — more accurate than markdown
            cl_files = sorted(Path(out_dir).rglob("*_content_list.json"))
            if cl_files:
                best_cl = max(cl_files, key=lambda p: p.stat().st_size)
                content_list = json.loads(best_cl.read_text(encoding="utf-8"))
                return self._extract_ref_blocks_with_pages(content_list)

            # Fallback: markdown (no page_idx, assign -1)
            md_files = sorted(Path(out_dir).rglob("*.md"))
            if not md_files:
                raise RuntimeError(
                    "MinerU completed but no output found (neither content_list.json nor *.md)"
                )
            best_md = max(md_files, key=lambda p: p.stat().st_size)
            refs = self.extract_references_from_markdown(
                best_md.read_text(encoding="utf-8", errors="ignore")
            )
            return [(r, -1) for r in refs]

    # ------------------------------------------------------------------
    # Public: extract markdown
    # ------------------------------------------------------------------

    async def extract_markdown(self, pdf_path: str) -> str:
        """Chạy MinerU và trả về markdown content."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        self._ensure_cli_on_path(self.command_template)
        if not self._command_exists(self.command_template):
            raise RuntimeError(
                "MinerU CLI not found in PATH. "
                f"Template: {self.command_template}"
            )

        with tempfile.TemporaryDirectory(prefix="mineru_out_") as out_dir:
            await self._run_mineru(pdf_path, out_dir)

            md_files = sorted(Path(out_dir).rglob("*.md"))
            if not md_files:
                raise RuntimeError("MinerU completed but no *.md output found")

            best_md = max(md_files, key=lambda p: p.stat().st_size)
            markdown_text = best_md.read_text(encoding="utf-8", errors="ignore")

            if self.debug_markdown_dir:
                self._save_debug_markdown(pdf_path, markdown_text)

            return markdown_text

    # ------------------------------------------------------------------
    # content_list.json extraction  ← main path
    # ------------------------------------------------------------------

    def _extract_ref_blocks_with_pages(
        self,
        content_list: list[dict],
    ) -> list[tuple[str, int]]:
        """
        Core extractor: return list (ref_text, page_idx).
        page_idx = -1 when no page info.
        """
        # Collect raw blocks with page_idx before splitting
        raw_blocks: list[tuple[str, int]] = []  # (text, page_idx)
        in_references = False

        for item in content_list:
            item_type = item.get("type", "text")

            if item_type in self._IGNORE_TYPES:
                continue

            text: str = item.get("text", "").strip()
            text_level = item.get("text_level")
            page_idx: int = item.get("page_idx", -1)

            is_heading = (
                item_type in self._HEADING_TYPES
                or text_level is not None
            )

            if is_heading:
                if not text:
                    continue
                heading_norm = re.sub(r"\s+", " ", text.lower()).strip("#").strip()

                if heading_norm in self.REFERENCE_HEADINGS:
                    in_references = True
                    continue

                if in_references:
                    is_major_heading = (
                        item_type in self._HEADING_TYPES
                        or (text_level is not None and text_level <= 3)
                    )
                    if is_major_heading and len(text) < 120:
                        looks_like_ref = bool(
                            re.search(r"\d{4}|doi:|https?://|et al\.|pp\.\s*\d", text, re.I)
                        )
                        if not looks_like_ref:
                            break
                continue

            if not in_references:
                continue

            if text and self._POST_REF_STOP.match(text) and len(text) < 100:
                break

            if item_type == "list":
                for li in item.get("list_items", []):
                    li_text = ""
                    if isinstance(li, str):
                        li_text = li.strip()
                    elif isinstance(li, dict):
                        li_text = li.get("text", "").strip()
                    if li_text:
                        raw_blocks.append((li_text, page_idx))
                continue

            if not text:
                continue

            if item_type in ("text", "list_item", "paragraph"):
                raw_blocks.append((text, page_idx))

        if not raw_blocks:
            return []

        # Split each block into individual refs, keep page_idx from the first block
        result: list[tuple[str, int]] = []
        for block_text, page_idx in raw_blocks:
            entries = self._split_content_list_refs(block_text)
            for e in entries:
                cleaned = self._normalize_reference_text(e)
                if len(cleaned) >= 15:
                    result.append((cleaned, page_idx))

        return result


    # ------------------------------------------------------------------
    # Markdown extraction  ← fallback path
    # ------------------------------------------------------------------

    def extract_references_from_markdown(self, markdown_text: str) -> List[str]:
        lines = markdown_text.splitlines()
        start_idx = self._find_references_start(lines)
        if start_idx is None:
            return []
        ref_lines = self._collect_references_block(lines, start_idx)
        if not ref_lines:
            return []
        block = "\n".join(ref_lines)
        parsed = self._split_content_list_refs(block)
        cleaned = [self._normalize_reference_text(x) for x in parsed]
        return [x for x in cleaned if len(x) >= 20]

    def _find_references_start(self, lines: List[str]) -> Optional[int]:
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            normalized = re.sub(r"\s+", " ", stripped.lower().strip("#").strip())
            if normalized in self.REFERENCE_HEADINGS:
                return i + 1
            if re.match(
                r"^#{1,6}\s*(references?|bibliography|works\s+cited)\s*$",
                stripped,
                flags=re.IGNORECASE,
            ):
                return i + 1
        return None

    def extract_references_block_markdown(self, markdown_text: str) -> str:
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
            if stripped.startswith("#"):
                if re.match(
                    r"^#{1,6}\s*(references?|bibliography|works\s+cited)\s*$",
                    stripped,
                    flags=re.IGNORECASE,
                ):
                    continue
                break
            if self._POST_REF_STOP.match(stripped):
                break
            ref_lines.append(line)
        return ref_lines

    # ------------------------------------------------------------------
    # Core splitter  ← dùng cho cả 2 paths
    # ------------------------------------------------------------------

    @classmethod
    def _split_content_list_refs(cls, text: str) -> list[str]:
        """
        Split reference block into individual entries.
        Supports: numbered [1]/1., author-year, mixed.

        Fix compared to old version:
        - Use 25% threshold (instead of any()) to classify numbered/author-year
          -> avoid a single line with a marker ruining the entire author-year block
        - Author-year mode uses _is_new_ref_start() with multi-signal scoring
          -> do not cut off continuation lines just because they end with "."
        """     
        numbered_pattern = re.compile(
            r"^\s*(?:\[[\w\d]+\]|\(\d+\)|\d{1,3}[.)]\s)\s*"
        )

        lines = text.splitlines()
        non_empty = [l for l in lines if l.strip()]

        # Need >= 25% lines with markers to be considered numbered mode
        if non_empty:
            marker_count = sum(1 for l in non_empty if numbered_pattern.match(l))
            is_numbered = (marker_count / len(non_empty)) >= 0.25
        else:
            is_numbered = False

        refs: list[str] = []
        buffer: list[str] = []

        def flush() -> None:
            if buffer:
                merged = re.sub(r"\s+", " ", " ".join(buffer)).strip()
                if merged:
                    refs.append(merged)
                buffer.clear()

        if is_numbered:
            # Numbered mode: split at marker, accumulate continuation lines
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue
                if numbered_pattern.match(line):
                    flush()
                    clean = numbered_pattern.sub("", line).strip()
                    if clean:
                        buffer.append(clean)
                else:
                    buffer.append(stripped)
            flush()

        else:
            # Author-year mode: blank line = hard boundary, else multi-signal
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    flush()
                    continue
                if buffer and _is_new_ref_start(stripped, buffer):
                    flush()
                buffer.append(stripped)
            flush()

        return refs

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_reference_text(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _save_debug_markdown(self, pdf_path: str, markdown_text: str) -> None:
        assert self.debug_markdown_dir
        os.makedirs(self.debug_markdown_dir, exist_ok=True)
        pdf_name = Path(pdf_path).stem

        debug_file = Path(self.debug_markdown_dir) / f"{pdf_name}.md"
        debug_file.write_text(markdown_text, encoding="utf-8")

        refs_block = self.extract_references_block_markdown(markdown_text)
        refs_file = Path(self.debug_markdown_dir) / f"{pdf_name}_references.md"
        refs_file.write_text(refs_block, encoding="utf-8")

        for extra in Path(self.debug_markdown_dir).glob(f"{pdf_name}*"):
            if extra.suffix not in {".md"}:
                try:
                    extra.unlink()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Page-boundary helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_page_boundary_pairs(
        blocks: list[tuple[str, int]],
    ) -> list[int]:
        """
        Return list of index i such that blocks[i] and blocks[i+1]
        are on different pages (page_idx different and both >= 0).
        """
        boundaries: list[int] = []
        for i in range(len(blocks) - 1):
            _, p1 = blocks[i]
            _, p2 = blocks[i + 1]
            if p1 >= 0 and p2 >= 0 and p1 != p2:
                boundaries.append(i)
        return boundaries

    @staticmethod
    def _should_skip_merge(text_a: str, text_b: str) -> bool:
        """
        Fast-reject: return True  if 2 blocks are definitely separate references

        Only use clear structural signals:
        - text_b starts with numbered citation marker ([1], 1., (1))
        """
        numbered = re.match(r"^\s*(?:\[\w+\]|\(\d+\)|\d{1,3}[.):]\s)", text_b)
        return bool(numbered)

    # ------------------------------------------------------------------
    # Debug helper
    # ------------------------------------------------------------------

    @classmethod
    def debug_ref_region(cls, content_list_path: str, window: int = 40) -> None:
        """
        In ra các blocks xung quanh khu vực References.
        Dùng để inspect MinerU gán type/text_level gì cho từng block.
        """
        data = json.loads(Path(content_list_path).read_text(encoding="utf-8"))
        for i, block in enumerate(data):
            text = block.get("text", "").strip()
            if re.search(r"reference|bibliography|tham kh[aả]o", text, re.I) and len(text) < 80:
                print(f"\n{'=' * 65}")
                print(f"Found at index {i}: [{block.get('type')}] level={block.get('text_level')} | '{text}'")
                print(f"{'=' * 65}")
                for j, b in enumerate(data[i: i + window]):
                    snippet = b.get("text", "")[:72].replace("\n", " ")
                    print(
                        f"  [{i+j:3d}] type={b.get('type','?'):18s} "
                        f"level={str(b.get('text_level','?')):4s} | {snippet}"
                    )
                break


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------

class AsyncLLMClient:
    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
        use_json_mode: bool = False,
    ):
        raw = AsyncOpenAI(api_key=api_key, base_url=base_url)
        if use_json_mode:
            self.client = instructor.from_openai(raw, mode=instructor.Mode.JSON)
        else:
            self.client = instructor.from_openai(raw)
        self.model = model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def decide_merge(self, text_a: str, text_b: str) -> bool:
        """
        Hỏi LLM xem text_a và text_b có phải là 2 phần của cùng một
        reference bị ngắt trang không.
        Trả về True nếu nên merge.
        """
        decision: MergeDecision = await self.client.chat.completions.create(
            model=self.model,
            response_model=MergeDecision,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a bibliographic citation expert. "
                        "Two text fragments were extracted from adjacent pages of a PDF. "
                        "Determine whether they are TWO SEPARATE references or "
                        "ONE reference split across pages.\n"
                        "Answer should_merge=true ONLY if fragment B is clearly a "
                        "continuation of fragment A (same bibliographic entry). "
                        "Answer should_merge=false if fragment B starts a new, "
                        "independent reference."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Fragment A (end of page):\n{text_a}\n\n"
                        f"Fragment B (start of next page):\n{text_b}"
                    ),
                },
            ],
        )
        return decision.should_merge

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def extract_citation(self, raw_text: str) -> ParsedCitationEntry:
        """Extract metadata for 1 raw citation string."""
        return await self.client.chat.completions.create(
            model=self.model,
            response_model=ParsedCitationEntry,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a high-precision citation extraction specialist.\n"
                        "Extract metadata from the provided raw text. "
                        "If information like DOI or arXiv ID is not present, leave it null.\n"
                        "Do not use external knowledge or invent data."
                    ),
                },
                {"role": "user", "content": f"raw_text: {raw_text}"},
            ],
        )

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def extract_citations_batch(self, raw_text: str) -> CitationCollection:
        """Extract multiple references from a block of text at once."""
        return await self.client.chat.completions.create(
            model=self.model,
            response_model=CitationCollection,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract each individual bibliographic citation from the text "
                        "into a structured collection."
                    ),
                },
                {"role": "user", "content": raw_text},
            ],
        )