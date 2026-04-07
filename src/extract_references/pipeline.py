"""
pipeline.py
The simplified orchestration pipeline using Instructor + LLM for structured extraction.
"""
import asyncio
from typing import List
from pathlib import Path
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
        """Full pipeline: PDF -> MinerU -> MD -> Extract Block -> Batch Split -> LLM -> JSON"""
        print(f"[Pipeline] Processing PDF: {pdf_path}...")
        ref_block = await self.mineru.extract_references_from_pdf(pdf_path)
        
        log_id = Path(pdf_path).stem[:10]
        return await self._process_block_to_results(ref_block, log_id)

    async def run_from_markdown(self, md_path: str) -> List[ExtractedCitation]:
        """Full pipeline: .md -> Extract Block -> Batch Split -> LLM -> JSON"""
        md_text = Path(md_path).read_text(encoding="utf-8", errors="ignore")
        
        print(f"[Pipeline] Extracting references block from {md_path}...")
        ref_block = self.mineru.extract_references_block_from_markdown(md_text)
        
        log_id = Path(md_path).stem[:10]
        return await self._process_block_to_results(ref_block, log_id)

    async def _process_block_to_results(self, ref_block: str, log_id: str) -> List[ExtractedCitation]:
        """Shared logic to batch-process a raw reference text block."""
        if not ref_block:
            print(f"[Pipeline][{log_id}] No references found.")
            return []

        # Split by double newlines as per user suggestion
        raw_items = [item.strip() for item in ref_block.split("\n\n") if item.strip()]
        
        # Batch items into groups (smaller batches for better reliability)
        batch_size = 5
        batches = [raw_items[i : i + batch_size] for i in range(0, len(raw_items), batch_size)]
        
        results_batches = []
        print(f"[Pipeline][{log_id}] Processing {len(raw_items)} citations in {len(batches)} batches via LLM...")
        
        # Run batches concurrently within semaphore limits
        tasks = [self._process_single_batch(i, batch) for i, batch in enumerate(batches, 1)]
        results_batches = await asyncio.gather(*tasks)

        return self._post_process_results(results_batches)

    async def _process_single_batch(self, batch_idx: int, batch_items: List[str]) -> List[ExtractedCitation]:
        async with self.semaphore:
            batch_text = "\n\n".join(batch_items)
            print(f"  [Batch {batch_idx}] Sending {len(batch_items)} citations to LLM...")
            try:
                collection = await self.llm.extract_citations_batch(batch_text)
                batch_results = []
                for parsed in collection.citations:
                    # Validation / Plausibility check
                    if not self.engine.is_plausible_reference(parsed.raw_text or "", parsed.model_dump()):
                         continue

                    batch_results.append(ExtractedCitation(
                        ref_id="TEMP", 
                        raw_text=parsed.raw_text or batch_text,
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
                return batch_results
            except Exception as e:
                print(f"  [Batch {batch_idx}] Failed: {e}")
                return []

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
