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
from .schemas import ParsedCitationEntry, CitationCollection


# ---------------------------------------------------------------------------
# Helpers (module-level để dễ test độc lập)
# ---------------------------------------------------------------------------

def _is_new_ref_start(line: str, current_buffer: list[str]) -> bool:
    """
    Multi-signal heuristic: quyết định `line` có bắt đầu một reference mới không.
    Yêu cầu tổng điểm >= 2 để tránh cắt nhầm continuation lines.
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

    Tối ưu performance:
    - Dùng command_template để inject các flags tắt formula/table/OCR
    - Ví dụ: "mineru -p {pdf} -o {out_dir} -b pipeline -m txt -d cpu -f false -t false"
    - Batch: gọi extract_raw_references() concurrently với asyncio.gather()
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

    # Sections thường xuất hiện SAU references → stop collecting
    _POST_REF_STOP = re.compile(
        r"^(appendix|acknowledg|author\s+(information|contribution|note)|"
        r"supplement|conflict\s+of\s+interest|funding|notes?\s*$|annex|"
        r"about\s+the\s+author|ethical\s+approval|declaration)",
        re.IGNORECASE,
    )

    # Types MinerU dùng cho noise — luôn bỏ qua
    _IGNORE_TYPES: frozenset[str] = frozenset({
        "page_number", "header", "page_footnote", "footer", "discarded",
    })

    # Types MinerU dùng cho headings
    _HEADING_TYPES: frozenset[str] = frozenset({
        "title", "section_header",
    })

    # Types chứa reference content
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
    async def extract_raw_references(self, pdf_path: str) -> List[str]:
        """Run MinerU on *pdf_path* và trả về list các reference string."""
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

            # Ưu tiên content_list.json — chính xác hơn markdown
            cl_files = sorted(Path(out_dir).rglob("*_content_list.json"))
            if cl_files:
                best_cl = max(cl_files, key=lambda p: p.stat().st_size)
                content_list = json.loads(best_cl.read_text(encoding="utf-8"))
                return self.extract_references_from_content_list(content_list)

            # Fallback: markdown
            md_files = sorted(Path(out_dir).rglob("*.md"))
            if not md_files:
                raise RuntimeError(
                    "MinerU completed but no output found (neither content_list.json nor *.md)"
                )
            best_md = max(md_files, key=lambda p: p.stat().st_size)
            return self.extract_references_from_markdown(
                best_md.read_text(encoding="utf-8", errors="ignore")
            )

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

    def extract_references_from_content_list(
        self,
        content_list: list[dict],
    ) -> list[str]:
        """
        Extract individual reference strings từ MinerU's content_list.json.

        Fixes so với version cũ:
        1. Detect heading bằng cả item_type VÀ text_level — không bỏ sót khi
           MinerU gán text_level=None cho "References" header.
        2. Break condition dùng item_type thay vì chỉ dựa text_level — tránh
           bỏ sót hoặc dư khi text_level thiếu nhất quán.
        3. Thu thập thêm list_item / paragraph types.
        4. _split_content_list_refs dùng threshold 25% thay vì any() để tránh
           false-positive numbered mode từ 1 dòng lẻ.
        """
        ref_blocks: list[str] = []
        in_references = False

        for item in content_list:
            item_type = item.get("type", "text")

            # Bỏ qua noise hoàn toàn
            if item_type in self._IGNORE_TYPES:
                continue

            text: str = item.get("text", "").strip()
            text_level = item.get("text_level")

            # ----------------------------------------------------------
            # Xác định đây có phải heading không
            # Điều kiện: type là heading type, HOẶC có text_level set
            # ----------------------------------------------------------
            is_heading = (
                item_type in self._HEADING_TYPES
                or text_level is not None
            )

            if is_heading:
                # Skip headings that have no useful text
                if not text:
                    continue
                heading_norm = re.sub(r"\s+", " ", text.lower()).strip("#").strip()

                # Tìm thấy References section
                if heading_norm in self.REFERENCE_HEADINGS:
                    in_references = True
                    continue

                # Đang trong References → quyết định dừng hay không
                if in_references:
                    # Chỉ dừng khi heading thực sự quan trọng:
                    # type là heading type HOẶC text_level cấp cao (<=3)
                    is_major_heading = (
                        item_type in self._HEADING_TYPES
                        or (text_level is not None and text_level <= 3)
                    )
                    if is_major_heading and len(text) < 120:
                        # Cho qua nếu text trông như reference (có năm, doi…)
                        looks_like_ref = bool(
                            re.search(r"\d{4}|doi:|https?://|et al\.|pp\.\s*\d", text, re.I)
                        )
                        if not looks_like_ref:
                            break
                # Heading nhưng chưa vào references → skip
                continue

            # ----------------------------------------------------------
            # Content block
            # ----------------------------------------------------------
            if not in_references:
                continue

            # Hard stop: sections xuất hiện sau references
            if text and self._POST_REF_STOP.match(text) and len(text) < 100:
                break

            # list blocks: text is empty, content lives in list_items
            if item_type == "list":
                for li in item.get("list_items", []):
                    if isinstance(li, str) and li.strip():
                        ref_blocks.append(li.strip())
                    elif isinstance(li, dict) and li.get("text", "").strip():
                        ref_blocks.append(li["text"].strip())
                continue

            # For all other content types, skip if no text
            if not text:
                continue

            if item_type == "text":
                ref_blocks.append(text)
            elif item_type in ("list_item", "paragraph"):
                ref_blocks.append(text)

        if not ref_blocks:
            return []

        # Join with double newlines to treat each block as a hard boundary for the splitter
        combined = "\n\n".join(ref_blocks)
        entries = self._split_content_list_refs(combined)
        cleaned = [self._normalize_reference_text(e) for e in entries]
        return [e for e in cleaned if len(e) >= 15]

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
        Tách reference block thành từng entry riêng lẻ.
        Hỗ trợ: numbered [1]/1., author-year, hỗn hợp.

        Fix so với version cũ:
        - Dùng threshold 25% (thay vì any()) để phân loại numbered/author-year
          → tránh 1 dòng lẻ có marker làm hỏng toàn bộ author-year block
        - Author-year mode dùng _is_new_ref_start() với multi-signal scoring
          → không cắt nhầm continuation lines chỉ vì kết thúc bằng "."
        """
        numbered_pattern = re.compile(
            r"^\s*(?:\[[\w\d]+\]|\(\d+\)|\d{1,3}[.)]\s)\s*"
        )

        lines = text.splitlines()
        non_empty = [l for l in lines if l.strip()]

        # Cần ≥ 25% dòng có marker mới coi là numbered mode
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
            # Numbered mode: split tại marker, accumulate continuation lines
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
    # Class-level convenience loader
    # ------------------------------------------------------------------

    @classmethod
    def load_and_extract_references(
        cls,
        content_list_path: str,
        output_path: str | None = None,
    ) -> list[str]:
        path = Path(content_list_path)
        content_list = json.loads(path.read_text(encoding="utf-8"))

        instance = cls()
        refs = instance.extract_references_from_content_list(content_list)

        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("\n\n".join(refs), encoding="utf-8")
            print(f"[MinerUClient] Wrote {len(refs)} references -> {out}")

        return refs

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
    ):
        self.client = instructor.from_openai(
            AsyncOpenAI(api_key=api_key, base_url=base_url)
        )
        self.model = model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def extract_citation(self, raw_text: str) -> ParsedCitationEntry:
        """Extract metadata cho 1 raw citation string."""
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
        """Extract nhiều references từ 1 block text cùng lúc."""
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