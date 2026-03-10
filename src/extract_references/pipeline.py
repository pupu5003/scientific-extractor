"""
pipeline.py
The master orchestration pipeline linking Extractor -> Heuristics -> LLM Validator.
"""
import asyncio
import aiohttp
from typing import List, Dict, Any
from .schemas import ExtractedCitation, ExtractedIdentifiers
from .clients import AsyncGrobidClient, AsyncLLMClient
from .heuristics import CitationParserEngine

class ExtractionPipeline:
    def __init__(self, grobid_url: str, llm_client: AsyncLLMClient, max_concurrency: int = 10):
        self.grobid = AsyncGrobidClient(grobid_url)
        self.llm = llm_client
        self.engine = CitationParserEngine()
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def run(self, pdf_path: str) -> List[ExtractedCitation]:
        print(f"[Pipeline] Extracting raw strings from {pdf_path}...")
        raw_strings = await self.grobid.extract_raw_references(pdf_path)
        print(f"[Pipeline] Found {len(raw_strings)} raw citations. Proceeding to structured parsing...")

        # Process all citations concurrently utilizing a shared connection pool
        async with aiohttp.ClientSession() as session:
            tasks = [self._process_single_citation(idx, raw, session) for idx, raw in enumerate(raw_strings, 1)]
            results = await asyncio.gather(*tasks)
            
        return [res for res in results if res is not None]

    async def _process_single_citation(self, idx: int, raw_text: str, session: aiohttp.ClientSession) -> ExtractedCitation:
        async with self.semaphore:
            # Step 1: Deterministic Parsing (GROBID)
            xml_content = await self.grobid.parse_citation_string(raw_text, session)
            parsed_dict = self.engine.digest_grobid_xml(raw_text, xml_content)
            
            # Step 2: Heuristic Fallbacks
            parsed_dict = self.engine.apply_regex_fallbacks(parsed_dict)
            
            # Step 3: LLM Confidence Routing
            needs_review = self._requires_llm_intervention(parsed_dict)
            llm_reviewed = False
            
            if needs_review:
                try:
                    patch = await self.llm.review_citation(raw_text, parsed_dict)
                    safe_fill = self.engine.guard_hallucinations(raw_text, patch.fill)
                    safe_corr = self.engine.guard_hallucinations(raw_text, patch.corrections)
                    
                    # Merge LLM patches
                    for k, v in safe_fill.items():
                        if not parsed_dict.get(k) and v:
                            parsed_dict[k] = v
                    parsed_dict.update(safe_corr)
                    llm_reviewed = True
                except Exception as e:
                    print(f"[{idx}] LLM Review Failed: {e}. Falling back to deterministic data.")

            # Step 4: Map to Strict Schema
            identifiers = ExtractedIdentifiers(
                doi=parsed_dict.get("doi"),
                arxiv_id=parsed_dict.get("arxiv_id"),
                url=parsed_dict.get("url")
            )
            
            return ExtractedCitation(
                ref_id=f"R{idx}",
                raw_text=raw_text,
                title=parsed_dict.get("title"),
                authors=parsed_dict.get("authors", []),
                venue=parsed_dict.get("venue"),
                year=parsed_dict.get("year"),
                identifiers=identifiers,
                llm_reviewed=llm_reviewed,
                confidence_score=0.8 if llm_reviewed else 1.0
            )

    def _requires_llm_intervention(self, parsed: Dict[str, Any]) -> bool:
        """Determines if the LLM should be invoked to fix missing critical data."""
        if not parsed.get("title") or len(parsed.get("title", "")) < 6: return True
        if not parsed.get("year"): return True
        if not parsed.get("authors"): return True
        raw = parsed.get("raw_text", "").lower()
        if "doi:" in raw and not parsed.get("doi"): return True
        if "arxiv" in raw and not parsed.get("arxiv_id"): return True
        return False