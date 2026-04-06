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
        """Full pipeline: PDF -> MinerU -> Raw Strings -> Instructor LLM -> JSON"""
        print(f"[Pipeline] Extracting raw strings from {pdf_path}...")
        raw_strings = await self.mineru.extract_raw_references(pdf_path)

        base_name = os.path.basename(pdf_path)
        log_id = base_name[:10]
        print(f"[Pipeline][{log_id}] Processing {len(raw_strings)} citations via LLM...")

        tasks = [self._process_single_citation(idx, raw) for idx, raw in enumerate(raw_strings, 1)]
        results = await asyncio.gather(*tasks)
        
        return self._post_process_results(results)

    async def run_from_content_list(self, content_list_path: str) -> List[ExtractedCitation]:
        """MinerU content_list.json -> Raw Strings -> Instructor LLM -> JSON"""
        import json
        from pathlib import Path
        path = Path(content_list_path).resolve()
        content_list = json.loads(path.read_text(encoding="utf-8"))

        raw_strings = self.mineru.extract_references_from_content_list(content_list)
        if not raw_strings:
            return []

        log_id = path.stem[:10]
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
                                raw_text=raw_text[:100] + "...", # truncate raw for batch items
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