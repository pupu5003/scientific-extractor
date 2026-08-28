"""
clients.py
Async clients for MinerU extraction and LLM providers.
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

import instructor
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .schemas import CitationCollection, ParagraphClaimCollection


# ---------------------------------------------------------------------------
# MinerU client
# ---------------------------------------------------------------------------

class AsyncMinerUClient:
    """Async wrapper around the MinerU CLI for PDF → Markdown conversion."""

    REFERENCE_HEADINGS: frozenset[str] = frozenset({
        "references",
        "reference",
        "bibliography",
        "works cited",
        "citations",
        "tai lieu tham khao",
        "tài liệu tham khảo",
    })

    # Sections that appear after references → stop collecting
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
        """Run MinerU CLI; raise RuntimeError on failure."""
        cmd = self.command_template.format(
            pdf=shlex.quote(pdf_path), out_dir=shlex.quote(out_dir)
        )
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
    # Public: PDF → Markdown
    # ------------------------------------------------------------------

    async def extract_markdown(self, pdf_path: str) -> str:
        """Run MinerU on *pdf_path* and return the full Markdown output."""
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

            md_files = sorted(Path(out_dir).rglob("*.md"))
            if not md_files:
                raise RuntimeError("MinerU completed but no *.md output found")

            best_md = max(md_files, key=lambda p: p.stat().st_size)
            markdown_text = best_md.read_text(encoding="utf-8", errors="ignore")

            if self.debug_markdown_dir:
                self._save_debug_markdown(pdf_path, markdown_text)

            return markdown_text

    # ------------------------------------------------------------------
    # Public: isolate References section
    # ------------------------------------------------------------------

    def extract_references_block_markdown(self, markdown_text: str) -> str:
        """Return only the 'References' section from a MinerU Markdown document."""
        lines = markdown_text.splitlines()
        start_idx = self._find_references_start(lines)
        if start_idx is None:
            return "# References\n"
        ref_lines = self._collect_references_block(lines, start_idx)
        block = "\n".join(ref_lines).strip()
        return "# References\n\n" + block + "\n"

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
    # Public: isolate body text (everything before References)
    # ------------------------------------------------------------------

    def extract_body_markdown(self, markdown_text: str) -> str:
        """Return the markdown content that precedes the References section.

        This is the text the claim-matching step (`claims.py`) searches for
        in-text citation markers ("[1]", "(Smith et al., 2020)", ...). If no
        References heading is found, the entire document is returned so
        claim-matching still has something to search.
        """
        lines = markdown_text.splitlines()
        start_idx = self._find_references_start(lines)
        if start_idx is None:
            return markdown_text
        heading_idx = start_idx - 1  # back up to the heading line itself
        return "\n".join(lines[:heading_idx]).strip()

    # ------------------------------------------------------------------
    # Debug helper
    # ------------------------------------------------------------------

    def _save_debug_markdown(self, pdf_path: str, markdown_text: str) -> None:
        assert self.debug_markdown_dir
        os.makedirs(self.debug_markdown_dir, exist_ok=True)
        pdf_name = Path(pdf_path).stem

        (Path(self.debug_markdown_dir) / f"{pdf_name}.md").write_text(
            markdown_text, encoding="utf-8"
        )
        refs_block = self.extract_references_block_markdown(markdown_text)
        (Path(self.debug_markdown_dir) / f"{pdf_name}_references.md").write_text(
            refs_block, encoding="utf-8"
        )
        for extra in Path(self.debug_markdown_dir).glob(f"{pdf_name}*"):
            if extra.suffix != ".md":
                try:
                    extra.unlink()
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# LLM client
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
        self.client = (
            instructor.from_openai(raw, mode=instructor.Mode.JSON)
            if use_json_mode
            else instructor.from_openai(raw)
        )
        self.model = model

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def extract_citations_batch(self, ref_block: str) -> CitationCollection:
        """
        Single-pass extraction: send the entire references block and receive a
        validated CitationCollection in one LLM call.

        The model is instructed to split any run-on or merged citations it
        encounters while building the JSON output.
        """
        return await self.client.chat.completions.create(
            model=self.model,
            response_model=CitationCollection,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a high-precision bibliographic citation extraction specialist.\n\n"
                        "INPUT: A raw references section copied verbatim from a scientific PDF. "
                        "It may contain line-break artifacts, merged entries (two papers run "
                        "together without a blank line), or irregular formatting.\n\n"
                        "YOUR TASK:\n"
                        "1. Identify every individual bibliographic entry, even when entries "
                        "are merged or concatenated. Split them implicitly as you produce the JSON.\n"
                        "2. For each entry extract:\n"
                        "   • raw_text  – the verbatim source text for that citation\n"
                        "   • title     – full paper/book/report title\n"
                        "   • authors   – list of 'FirstName LastName' strings\n"
                        "   • venue     – journal, conference series, arXiv, or repository name\n"
                        "   • year      – 4-digit publication year\n"
                        "   • doi       – DOI string if present (omit 'https://doi.org/' prefix)\n"
                        "   • arxiv_id  – arXiv identifier if present (e.g. '2402.12345')\n"
                        "   • url       – URL only when no DOI or arXiv ID is available\n"
                        "3. NEVER invent data. Leave a field null when the information is absent "
                        "from the source text.\n"
                        "4. Output one CitationCollection object containing one entry per unique paper."
                    ),
                },
                {
                    "role": "user",
                    "content": f"References block:\n\n{ref_block}",
                },
            ],
        )


    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def extract_claims_batch(self, tagged_paragraph: str) -> ParagraphClaimCollection:
        """
        Extract atomic claims from one paragraph of body text.

        The paragraph has already been pre-processed (see `claims.py` /
        `claim_extraction.py`): in-text citation markers resolved to
        canonical '<CIT:ref_id>' tags, and sentences numbered
        '[S1] ... [S2] ...'. The model never has to understand APA/IEEE/etc
        citation syntax — it only ever sees '<CIT:ref_id>' tokens.
        """
        return await self.client.chat.completions.create(
            model=self.model,
            response_model=ParagraphClaimCollection,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise scientific claim extraction specialist.\n\n"
                        "INPUT: One paragraph from a scientific paper. Sentences are "
                        "tagged '[S<n>]' at their start. Some sentences contain inline "
                        "tokens like '<CIT:R3>' marking exactly where an in-text "
                        "citation to reference R3 occurs.\n\n"
                        "YOUR TASK: Extract the atomic factual claims made in this "
                        "paragraph. For each claim, report:\n"
                        "  • claim               – a self-contained statement of the claim, "
                        "in your own words if needed, but never inventing facts not in the text\n"
                        "  • source_sentence_ids – the [S<n>] id(s) the claim is drawn from\n"
                        "  • explicit_citations  – ref ids (e.g. 'R3') whose <CIT:...> tag "
                        "appears directly in the claim's own source sentence(s)\n"
                        "  • inherited_citations – ref ids that are NOT tagged in the "
                        "claim's own sentence(s), but that the claim clearly still refers "
                        "to via discourse (a pronoun, 'this approach', 'these methods', a "
                        "repeated short-form name, etc.) pointing back to an earlier "
                        "sentence in THIS SAME paragraph that WAS tagged with that ref id\n\n"
                        "RULES:\n"
                        "1. NEVER invent a ref id — only use ids that literally appear as "
                        "'<CIT:...>' somewhere in this paragraph.\n"
                        "2. A claim with no citation at all (pure background/motivation) is "
                        "fine — leave both citation lists empty.\n"
                        "3. Do not merge unrelated claims from different citations into one; "
                        "split them so each claim maps to the citation(s) that actually "
                        "support it, not every citation in the paragraph.\n"
                        "4. Skip meta-text that isn't a factual claim (e.g. pure section "
                        "transitions)."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Paragraph:\n\n{tagged_paragraph}",
                },
            ],
        )
