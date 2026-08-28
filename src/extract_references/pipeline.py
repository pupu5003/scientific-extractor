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
  5. Claim extraction        – Deterministic in-text citation resolution
                               (see `claims.py`) + a paragraph-level LLM
                               pass (see `claim_extraction.py`) that pulls
                               out atomic claims and links each one back to
                               the reference(s) it cites, distinguishing
                               explicit citations from ones only implied by
                               discourse within the same paragraph.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from .clients import AsyncLLMClient, AsyncMinerUClient
from .heuristics import CitationParserEngine
from .claim_extraction import attach_claim_ids, extract_claims_for_document
from .schemas import (
    Claim,
    CitationCollection,
    ExtractedCitation,
    ExtractedIdentifiers,
    ExtractionResult,
)


class ExtractionPipeline:
    def __init__(
        self,
        mineru_cmd: str,
        llm_client: AsyncLLMClient,
        max_concurrency: int = 10,  # also used as the claim-extraction paragraph concurrency
        debug_markdown_dir: str | None = None,
        skip_claims: bool = False,
    ):
        self.mineru = AsyncMinerUClient(mineru_cmd, debug_markdown_dir=debug_markdown_dir)
        self.llm = llm_client
        self.engine = CitationParserEngine()
        self.max_concurrency = max_concurrency
        self.skip_claims = skip_claims

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self, pdf_path: str) -> ExtractionResult:
        """
        End-to-end pipeline for a single PDF.

        Returns an `ExtractionResult` holding the re-indexed reference list
        and the claims extracted from the body text (empty if
        `skip_claims=True` or no References section was found).
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
            return ExtractionResult(references=[], claims=[])

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
            return ExtractionResult(references=[], claims=[])

        # Step 4 – Validate and map to output schema
        results = self._build_citations(collection)
        print(f"[Pipeline][{log_id}] Step 4: Built {len(results)} citation(s).")

        # Step 5 – Extract claims from the body text and link them to references
        claims: List[Claim] = []
        if not self.skip_claims and results:
            print(f"[Pipeline][{log_id}] Step 5: Extracting claims from body text...")
            body_markdown = self.mineru.extract_body_markdown(markdown)
            try:
                claims = await extract_claims_for_document(
                    self.llm, results, body_markdown, max_concurrency=self.max_concurrency
                )
                attach_claim_ids(results, claims)
            except Exception as exc:
                print(f"[Pipeline][{log_id}] Claim extraction failed: {exc}")

        print(
            f"[Pipeline][{log_id}] Done — {len(results)} citation(s), "
            f"{len(claims)} claim(s) extracted."
        )
        return ExtractionResult(references=results, claims=claims)

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
