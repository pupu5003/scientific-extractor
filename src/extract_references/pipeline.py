"""
pipeline.py
The simplified orchestration pipeline using Instructor + LLM for structured extraction.
"""
import asyncio
import os
from typing import List
from .schemas import ExtractedCitation, ExtractedIdentifiers
from .clients import AsyncMinerUClient, AsyncLLMClient
from .heuristics import CitationParserEngine
from pathlib import Path

class ExtractionPipeline:
    def __init__(
        self,
        mineru_cmd: str,
        llm_client: AsyncLLMClient,
        max_concurrency: int = 10,
        debug_markdown_dir: str | None = None,
    ):
        self.mineru = AsyncMinerUClient(mineru_cmd, debug_markdown_dir=debug_markdown_dir)
        self.llm = llm_client
        self.engine = CitationParserEngine()
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def run(self, pdf_path: str) -> List[ExtractedCitation]:
        """Full pipeline: PDF -> MinerU -> Raw Strings -> (page-split resolve) -> LLM -> JSON"""
        print(f"[Pipeline] Extracting raw strings from {pdf_path}...")
        raw_blocks = await self.mineru.extract_ref_blocks_with_page_idx(pdf_path)

        log_id = Path(pdf_path).stem[:10]

        # Resolve references split across page boundaries
        raw_strings = await self._resolve_page_splits(raw_blocks, log_id)

        return await self.process_citations(raw_strings, log_id)

    async def _resolve_page_splits(
        self,
        blocks: list[tuple[str, int]],
        log_id: str = "SPLIT",
    ) -> List[str]:
        """
        Detect reference split across page boundaries and ask LLM if two references should be merged.

        Logic:
        1. Find indices where page_idx differs between adjacent blocks.
        2. Fast-reject: if text_b starts with a numbered marker, they are separate.
        3. The remaining pairs -> ask LLM in parallel (protected by semaphore).
        4. If should merge -> merge text_a + " " + text_b into one entry.
        """

        if not blocks:
            return []

        boundary_indices = AsyncMinerUClient._detect_page_boundary_pairs(blocks)

        if not boundary_indices:
            print(f"[Pipeline][{log_id}] No page boundaries detected, skipping split resolution.")
            return [text for text, _ in blocks]

        print(
            f"[Pipeline][{log_id}] Found {len(boundary_indices)} page boundary pair(s), "
            "resolving with LLM..."
        )

        # Design: use set to track indices that have been merged (absorbed into previous)
        merged_into_prev: set[int] = set()

        async def _check_pair(i: int) -> bool:
            """Return True if blocks[i] and blocks[i+1] should be merged."""
            text_a, _ = blocks[i]
            text_b, _ = blocks[i + 1]

            # Fast-reject: text_b has numbered marker
            if AsyncMinerUClient._should_skip_merge(text_a, text_b):
                # print(
                #     f"[Pipeline][{log_id}] Boundary @{i}: fast-reject "
                #     f"(text_b has numbered marker)"
                # )
                return False

            # Call LLM (protected by semaphore)
            async with self.semaphore:
                try:
                    should_merge = await self.llm.decide_merge(text_a, text_b)
                    # print(
                    #     f"[Pipeline][{log_id}] Boundary @{i}: "
                    #     f"LLM says {'MERGE' if should_merge else 'KEEP SEPARATE'}"
                    # )
                    return should_merge
                except Exception as e:
                    print(f"[Pipeline][{log_id}] Boundary @{i}: decide_merge failed ({e}), keeping separate")
                    return False

        # Run all boundary checks in parallel
        merge_flags = await asyncio.gather(*[_check_pair(i) for i in boundary_indices])

        for idx, should_merge in zip(boundary_indices, merge_flags):
            if should_merge:
                merged_into_prev.add(idx + 1)

        # Build output list, merge when needed
        result: List[str] = []
        i = 0
        while i < len(blocks):
            if i in merged_into_prev:
                # Already merged into previous block - continue to the last entry
                if result:
                    result[-1] = result[-1].rstrip() + " " + blocks[i][0].lstrip()
                else:
                    result.append(blocks[i][0])
            else:
                result.append(blocks[i][0])
            i += 1

        merged_count = len([f for f in merge_flags if f])
        print(
            f"[Pipeline][{log_id}] Page-split resolution done: "
            f"{merged_count} merge(s), {len(result)} refs remaining."
        )
        return result

    async def process_citations(self, raw_strings: List[str], log_id: str = "EXTRACT") -> List[ExtractedCitation]:
        """Process a list of raw citation strings through the LLM."""
        if not raw_strings:
            return []

        print(f"[Pipeline][{log_id}] Processing {len(raw_strings)} citations via LLM...")
        tasks = [self._process_single_citation(idx, raw) for idx, raw in enumerate(raw_strings, 1)]
        results = await asyncio.gather(*tasks)
        
        return self._post_process_results(results)


    def _post_process_results(self, results: List[List[ExtractedCitation]]) -> List[ExtractedCitation]:
        """Flatten results and re-index references (R1, R2...)."""
        final_results = []
        current_idx = 1
        for batch in results:
            for res in batch:
                res.ref_id = f"R{current_idx}"
                final_results.append(res)
                current_idx += 1
        return final_results

    async def _process_single_citation(self, idx: int, raw_text: str) -> List[ExtractedCitation]:
        """Process a raw string. Returns a LIST of citations (usually 1, but multiple if batch-split)."""
        async with self.semaphore:
            # 1. Clean raw text (deterministic)
            raw_text = self.engine.heal_broken_urls(raw_text)
            
            # Heuristic: If string is extremely long, it's likely multiple citations merged.
            # Use batch extraction instead of single.
            if len(raw_text) > 1000:
                print(f"[{idx}] Raw text very long ({len(raw_text)} chars), using batch extraction...")
                try:
                    collection = await self.llm.extract_citations_batch(raw_text)
                    results = []
                    for i, parsed in enumerate(collection.citations, 1):
                        if self.engine.is_plausible_reference(raw_text, parsed.model_dump()):
                            results.append(ExtractedCitation(
                                ref_id=f"R{idx}_{i}",
                                raw_text=raw_text, # Keep full original text for batch items
                                title=parsed.title,
                                authors=parsed.authors,
                                venue=parsed.venue,
                                year=parsed.year,
                                identifiers=ExtractedIdentifiers(
                                    doi=parsed.doi,
                                    arxiv_id=parsed.arxiv_id,
                                    url=parsed.url
                                )
                            ))
                    return results
                except Exception as e:
                    print(f"[{idx}] Batch Extraction Failed: {e}")
                    # Fallback to single extraction attempt

            try:
                # 2. Extract structured metadata using Instructor + LLM
                parsed = await self.llm.extract_citation(raw_text)
                
                # 3. Validation / Plausibility check
                parsed_dict = parsed.model_dump()
                if not self.engine.is_plausible_reference(raw_text, parsed_dict):
                    print(f"[{idx}] Skipping citation: low plausibility (Title found: {bool(parsed.title)})")
                    return []

                # 4. Map to final Schema
                return [ExtractedCitation(
                    ref_id=f"R{idx}",
                    raw_text=raw_text,
                    title=parsed.title,
                    authors=parsed.authors,
                    venue=parsed.venue,
                    year=parsed.year,
                    identifiers=ExtractedIdentifiers(
                        doi=parsed.doi,
                        arxiv_id=parsed.arxiv_id,
                        url=parsed.url
                    )
                )]
            except Exception as e:
                print(f"[{idx}] Extraction Failed: {e}")
                return []