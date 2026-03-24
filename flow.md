# Extraction Flow (Input PDF → Output JSON)

This document describes the execution flow from input PDF to output JSON in the reference extractor.

## 1) CLI Entry Point
- File: `src/extract_references/__main__.py`
- Entry: `python -m src.extract_references <pdf_path> [--grobid_url ...] [--llm_backend ...]`

Flow:
1. Parse CLI args.
2. Initialize `AsyncLLMClient`.
3. Create `ExtractionPipeline`.
4. Run:
   - `pipeline.run(pdf_path)` → reference list output (`*_extracted.json`)

Outputs:
- `tests/json/neurips2025/<pdf_name>_extracted.json`

---

## 2) Reference List Extraction (Reference Section)
- File: `src/extract_references/pipeline.py`
- Method: `ExtractionPipeline.run(pdf_path)`

Steps:
1. **GROBID reference list**
   - `AsyncGrobidClient.extract_raw_references(pdf_path)`
   - Endpoint: `/api/processReferences`
   - Returns raw reference strings.

2. **Per-reference parsing** (concurrent)
   - For each raw string → `_process_single_citation(idx, raw_text, session)`

3. **Structured parsing (anystyle)**
   - `AnystyleClient.parse(raw_text)`
   - Calls `anystyle -f json parse -` via async subprocess (stdin → stdout)
   - Returns parsed JSON dict for one citation.

4. **JSON digestion**
   - `CitationParserEngine.digest_anystyle_json(raw_text, anystyle_result)`
   - Maps anystyle fields → internal dict: `title`, `authors`, `venue`, `year`, `doi`, `url`.

5. **Regex fallbacks**
   - `CitationParserEngine.apply_regex_fallbacks(parsed_dict)`
   - Fills missing fields (doi/arxiv/url/year/venue and regex authors).

6. **LLM review (optional)**
   - `ExtractionPipeline._requires_llm_intervention(parsed_dict)`
   - If needed → `AsyncLLMClient.review_citation(raw_text, parsed_dict)`
   - Patch guarded via `CitationParserEngine.guard_hallucinations`.

7. **Filter non-references**
   - `CitationParserEngine.is_plausible_reference(raw_text, parsed_dict)`
   - Keeps only items with at least 2 key fields.

8. **Build output schema**
   - `ExtractedCitation` → appended to output list.

---

## 3) In-text Citation Contexts (Body Text)
- File: `src/extract_references/pipeline.py`
- Method: `ExtractionPipeline.extract_document_contexts(pdf_path)`

Steps:
1. **GROBID full text**
   - `AsyncGrobidClient.extract_fulltext_tei(pdf_path)`
   - Endpoint: `/api/processFulltextDocument`

2. **Extract contexts**
   - `CitationParserEngine.extract_document_contexts(tei_xml, source_document_id)`

3. **Map in-text marker → full reference string**
   - Builds `biblStruct` map from TEI
   - Resolves `<ref type="bibr" target="#bX">` → raw reference string

4. **Claim and surrounding context**
   - `claim_text`: sentence (`<s>`) containing the citation marker
   - `surrounding_context`: full paragraph (`<p>`)

5. **Coordinates (optional)**
   - If TEI contains `coords`, parsed into `VisualCoordinate` objects.

Output:
- List of `DocumentContext` items → saved to `*_contexts.json`.

---

## 4) Output Schemas
- `ExtractedCitation` (reference list)
- `DocumentContext` (in-text citation context)

Defined in `src/extract_references/schemas.py`.
