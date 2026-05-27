"""
pipeline.py
Single-pass extraction pipeline.

Flow:
  1. Layout-aware ingestion  – MinerU converts PDF → Markdown
  2. Section isolation       – Heuristic boundary detection extracts the
                               raw "References" block from the Markdown
  3. Structured generation   – ONE LLM call (instructor + Pydantic) turns
                               the whole block into a CitationCollection,
                               implicitly splitting any merged entries
  4. Validation & mapping    – Plausibility filter, then flatten to
                               List[ExtractedCitation]
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from .clients import AsyncLLMClient, AsyncMinerUClient
from .heuristics import CitationParserEngine
from .schemas import CitationCollection, ExtractedCitation, ExtractedIdentifiers


class ExtractionPipeline:
    def __init__(
        self,
        mineru_cmd: str,
        llm_client: AsyncLLMClient,
        max_concurrency: int = 10,  # kept for API compat with batch runner
        debug_markdown_dir: str | None = None,
    ):
        self.mineru = AsyncMinerUClient(mineru_cmd, debug_markdown_dir=debug_markdown_dir)
        self.llm = llm_client
        self.engine = CitationParserEngine()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self, pdf_path: str) -> List[ExtractedCitation]:
        """
        End-to-end pipeline for a single PDF.

        Returns a re-indexed list of ExtractedCitation objects, one per
        unique bibliographic entry found in the document.
        """
        log_id = Path(pdf_path).stem[:20]

        # Step 1 – Layout-aware ingestion
        print(f"[Pipeline][{log_id}] Step 1: PDF → Markdown (MinerU)...")
        markdown = await self.mineru.extract_markdown(pdf_path)

        # Step 2 – Isolate the References section
        print(f"[Pipeline][{log_id}] Step 2: Isolating 'References' section...")
        ref_block = self.mineru.extract_references_block_markdown(markdown)

        if not ref_block.strip() or ref_block.strip() == "# References":
            print(f"[Pipeline][{log_id}] No references section found — aborting.")
            return []

        # Pre-process: fix URL/DOI fragmentation from PDF line-breaks
        ref_block = self.engine.heal_broken_urls(ref_block)

        # Step 3 – Single-pass structured generation
        print(
            f"[Pipeline][{log_id}] Step 3: Single-pass LLM extraction "
            f"({len(ref_block):,} chars)..."
        )
        try:
            collection: CitationCollection = await self.llm.extract_citations_batch(ref_block)
        except Exception as exc:
            print(f"[Pipeline][{log_id}] LLM extraction failed: {exc}")
            return []

        # Step 4 – Validate and map to output schema
        results = self._build_citations(collection)
        print(f"[Pipeline][{log_id}] Done — {len(results)} citation(s) extracted.")
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_citations(self, collection: CitationCollection) -> List[ExtractedCitation]:
        """
        Convert each ParsedCitationEntry to an ExtractedCitation.

        Entries that fail the plausibility heuristic (e.g. LLM hallucinations
        with no meaningful fields) are silently dropped.
        """
        results: List[ExtractedCitation] = []
        for idx, parsed in enumerate(collection.citations, 1):
            anchor_text = parsed.raw_text or parsed.title or ""
            if not self.engine.is_plausible_reference(anchor_text, parsed.model_dump()):
                continue

            results.append(
                ExtractedCitation(
                    ref_id=f"R{idx}",
                    raw_text=parsed.raw_text,
                    title=parsed.title,
                    authors=parsed.authors,
                    venue=parsed.venue,
                    year=parsed.year,
                    identifiers=ExtractedIdentifiers(
                        doi=parsed.doi,
                        arxiv_id=parsed.arxiv_id,
                        url=parsed.url,
                    ),
                )
            )

        # Re-index to guarantee contiguous R1, R2, … numbering after filtering
        for new_idx, citation in enumerate(results, 1):
            citation.ref_id = f"R{new_idx}"

        return results
