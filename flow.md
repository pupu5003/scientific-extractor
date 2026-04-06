# Extraction Flow (Input PDF → Output JSON)

This document describes the high-precision extraction flow using **MinerU** and **LLM (Instructor)**.

## 1) Execution Entry Point
- **File**: `src/extract_references/__main__.py`
- **Logic**: 
  - If input is a **PDF**: Runs MinerU (`magic-pdf`) → Extracts content from structured JSON.
  - If input is a **Folder**: Auto-finds `content_list.json` in MinerU's output directory.

## 2) Content Extraction & Section Detection
- **File**: `src/extract_references/clients.py`
- **Method**: `AsyncMinerUClient.extract_references_from_content_list`
- **Steps**:
  1. Detect `References` or `Bibliography` heading.
  2. Collect all following blocks of `type: "text"` or `type: "list"`.
  3. Filter out noise (Footnotes, Headers, Page Numbers).
  4. Perform **Smart Splitting**:
     - Uses deterministic heuristics to identify record boundaries.
     - Handles papers without markers (e.g., ICLR/NeurIPS).

## 3) Structured LLM Parsing
- **File**: `src/extract_references/pipeline.py`
- **Method**: `ExtractionPipeline._process_single_citation`
- **Logic**:
  1. **Standard Mode**: Sends a single raw string to `AsyncLLMClient.extract_citation`.
  2. **Batch Mode Fallback**: If the raw string is suspiciously long (> 1000 chars), it calls `extract_citations_batch` for structural multi-citation parsing.
- **Tools**:
  - **Instructor (Pydantic)**: Enforces a strict schema (`ParsedCitationEntry`) directly from the LLM response.

## 4) Post-Processing & Validation
- **File**: `src/extract_references/pipeline.py`
- **Logic**:
  1. Filter out non-citations using `CitationParserEngine.is_plausible_reference`.
  2. Flatten results from batch calls.
  3. Re-index all citations (R1, R2, ...) for consistency.
  4. Save to `tests/json/minerU/<pdf_name>_extracted.json`.

## 5) Output Schema
Defined in `src/extract_references/schemas.py`:
- `ExtractedCitation`: Final structured output including `identifiers` (DOI, arXiv ID, URL).
