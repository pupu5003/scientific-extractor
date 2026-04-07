"""
clients.py
Lightweight async clients for MinerU (PDF -> MD) and LLM (Instructor).
"""
from __future__ import annotations
import asyncio
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


class AsyncMinerUClient:
    """
    Async client wrapping MinerU CLI to convert PDF to Markdown.
    Focuses on extracting the raw References block from the generated MD.
    """

    REFERENCE_HEADINGS: frozenset[str] = frozenset({
        "references", "reference", "bibliography", "works cited", 
        "citations", "tai lieu tham khao", "tài liệu tham khảo",
    })

    _POST_REF_STOP = re.compile(
        r"^(appendix|acknowledg|author\s+(information|contribution|note)|"
        r"supplement|conflict\s+of\s+interest|funding|notes?\s*$|annex|"
        r"about\s+the\s+author|ethical\s+approval|declaration)",
        re.IGNORECASE,
    )

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

    async def extract_references_from_pdf(self, pdf_path: str) -> str:
        """PDF -> MinerU -> Extract References Block from MD."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        self._ensure_cli_on_path()
        
        with tempfile.TemporaryDirectory(prefix="mineru_out_") as out_dir:
            await self._run_mineru(pdf_path, out_dir)
            
            md_files = sorted(Path(out_dir).rglob("*.md"))
            if not md_files:
                raise RuntimeError("MinerU failed to produce Markdown output.")
            
            best_md = max(md_files, key=lambda p: p.stat().st_size)
            markdown_text = best_md.read_text(encoding="utf-8", errors="ignore")
            
            if self.debug_markdown_dir:
                self._save_debug(pdf_path, markdown_text)
                
            return self.extract_references_block_from_markdown(markdown_text)

    def extract_references_block_from_markdown(self, markdown_text: str) -> str:
        """Finds the References section in MD and returns it as a raw block."""
        lines = markdown_text.splitlines()
        start_idx = self._find_references_start(lines)
        if start_idx is None:
            return ""
            
        ref_lines = []
        for line in lines[start_idx:]:
            stripped = line.strip()
            if not stripped:
                ref_lines.append("")
                continue
            
            if stripped.startswith("#"):
                heading_norm = re.sub(r"\s+", " ", stripped.lower().strip("#").strip())
                if heading_norm not in self.REFERENCE_HEADINGS:
                    break
            
            if self._POST_REF_STOP.match(stripped) and len(stripped) < 100:
                break
                
            ref_lines.append(line)
            
        return "\n".join(ref_lines).strip()

    def _find_references_start(self, lines: List[str]) -> Optional[int]:
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped: continue
            norm = re.sub(r"\s+", " ", stripped.lower().strip("#").strip())
            if norm in self.REFERENCE_HEADINGS or re.match(r"^#{1,6}\s*(references?|bibliography|works\s+cited)\s*$", stripped, re.I):
                return i + 1
        return None

    async def _run_mineru(self, pdf_path: str, out_dir: str) -> None:
        cmd = self.command_template.format(pdf=shlex.quote(pdf_path), out_dir=shlex.quote(out_dir))
        print(f"[MinerU] Running: {cmd}")
        proc = await asyncio.create_subprocess_shell(cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)
        if proc.returncode != 0:
            raise RuntimeError(f"MinerU failed: {stderr.decode('utf-8', errors='ignore')}")

    def _ensure_cli_on_path(self) -> None:
        first_token = shlex.split(self.command_template)[0]
        if shutil.which(first_token) is None:
            venv_bin = Path(sys.executable).resolve().parent
            os.environ["PATH"] = f"{venv_bin}{os.pathsep}{os.environ.get('PATH', '')}"

    def _save_debug(self, pdf_path: str, text: str) -> None:
        os.makedirs(self.debug_markdown_dir, exist_ok=True)
        debug_file = Path(self.debug_markdown_dir) / f"{Path(pdf_path).stem}.md"
        debug_file.write_text(text, encoding="utf-8")


class AsyncLLMClient:
    def __init__(self, api_key: str, base_url: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.client = instructor.from_openai(AsyncOpenAI(api_key=api_key, base_url=base_url))
        self.model = model

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def extract_citations_batch(self, raw_text: str) -> CitationCollection:
        """Extract multi-references from a text block via Instructor."""
        return await self.client.chat.completions.create(
            model=self.model,
            response_model=CitationCollection,
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "Extract each individual bibliographic citation from the text into a structured collection. "
                        "For each entry:\n"
                        "1. Copy the original raw text into the 'raw_text' field exactly as it appears.\n"
                        "2. Clean and rejoin URLs/DOIs that are broken by spaces or line breaks (e.g., 'https : //' -> 'https://').\n"
                        "3. Do not use external knowledge or invent data."
                    )
                },
                {"role": "user", "content": raw_text},
            ],
        )