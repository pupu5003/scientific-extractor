# Scientific Reference Extractor

A high-precision pipeline for extracting structured bibliographic references from scientific PDFs using **MinerU** and an **LLM** (via `instructor`).

---

## How it works

1. **MinerU** converts the PDF to Markdown.
2. The "References" block is located with deterministic heuristics and isolated from the rest of the document.
3. The whole block is parsed in a **single LLM call** (`instructor` + Pydantic) into a structured schema (`CitationCollection`), implicitly splitting any entries that got merged together.
4. Non-plausible entries are filtered out and the survivors are re-indexed to `R1, R2, ...`.
5. In-text citation markers (`[1]`, `(Smith et al., 2020)`, ...) in the body text are resolved against the reference list, then a per-paragraph LLM pass extracts atomic **claims** and links each one back to the reference(s) it cites. Skip this step with `--no_claims`.
6. Results are saved as one JSON file — see [Output format](#output-format).

---

## Prerequisites

- Python 3.10+
- [MinerU](https://github.com/opendatalab/MinerU) (`magic-pdf` / `mineru` CLI)
- An LLM API key (OpenAI, Together AI, or local Ollama)

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:

```env
OPENAI_API_KEY=sk-...
# or for Together AI:
TOGETHER_API_KEY=...
```

Download MinerU models (one-time):

```bash
.venv/bin/mineru-models-download
```

---

## Usage

### Run on a single PDF

```bash
python3 -m src.extract_references <path/to/paper.pdf> [options]
```

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--llm_backend` | `openai` | LLM provider: `openai`, `together`, `ollama` |
| `--model` | `gpt-4o-mini` | Model name |
| `--concurrency` | `10` | Max citations parsed concurrently, and max claim-extraction paragraphs parsed concurrently |
| `--output_dir` | `tests/json/minerU/` | Directory to save the output JSON |
| `--no_claims` | *(off)* | Skip LLM-based claim extraction from the body text — references only |
| `--mineru_cmd` | *(see below)* | MinerU command template |

**Examples:**

```bash
# OpenAI (default)
python3 -m src.extract_references tests/paper.pdf
python3 -m src.extract_references paper.pdf --llm_backend openai --model gpt-4o-mini

# Together AI
python3 -m src.extract_references tests/paper.pdf \
    --llm_backend together \
    --model meta-llama/Llama-3.3-70B-Instruct-Turbo \
    --output_dir out/

# Ollama (local)
python3 -m src.extract_references tests/paper.pdf \
    --llm_backend ollama \
    --model llama3
```

Output is saved as `<output_dir>/<pdf_stem>_extracted.json`.

---

### Run in batch mode

Process an entire folder (or glob pattern) of PDFs concurrently:

```bash
python3 run_batch.py <glob_or_dir> [options]
```



**Options:**

| Flag | Default | Description |
|---|---|---|
| `--output_dir` | `tests/json/batch_output/` | Directory to save all output JSON files |
| `--llm_backend` | `openai` | LLM provider: `openai`, `together`, `ollama` |
| `--model` | `gpt-4o-mini` | Model name |
| `--pdf_workers` | `5` | Max PDFs processed concurrently |
| `--citation_workers` | `10` | Max citations parsed concurrently per PDF, and max claim-extraction paragraphs parsed concurrently per PDF |
| `--no_claims` | *(off)* | Skip LLM-based claim extraction from the body text — references only |
| `--mineru_cmd` | *(see below)* | MinerU command template |
| `--debug_markdown_dir` | *(empty)* | If set, saves MinerU markdown output here for inspection |

**Examples:**

```bash
# Process all PDFs in a folder
python3 run_batch.py tests/pdfs/ --output_dir out/
python3 run_batch.py "tests/pdfs/iclr2025/*.pdf" --llm_backend openai --model gpt-4o-mini --output_dir tests/json/output/

# Use a glob pattern with Together AI
python3 run_batch.py "tests/pdfs/iclr2025/*.pdf" \
    --llm_backend together \
    --model meta-llama/Llama-3.3-70B-Instruct-Turbo \
    --pdf_workers 3 \
    --output_dir out/

# Save MinerU markdown for debugging
python3 run_batch.py tests/pdfs/ --debug_markdown_dir out/md_debug/
```

Each PDF produces one `<pdf_stem>_extracted.json` file in `--output_dir`.

---

## Output format

```json
{
  "references": [
    {
      "ref_id": "R1",
      "raw_text": "Vaswani et al. Attention is all you need. NeurIPS 2017.",
      "title": "Attention Is All You Need",
      "authors": ["Ashish Vaswani", "Noam Shazeer"],
      "venue": "NeurIPS 2017",
      "year": 2017,
      "identifiers": {
        "arxiv_id": "1706.03762",
        "url": "https://arxiv.org/abs/1706.03762"
      },
      "claim_ids": ["claim_001"]
    }
  ],
  "claims": [
    {
      "claim_id": "claim_001",
      "claim": "The Transformer architecture was introduced by Vaswani et al.",
      "source_sentence_ids": ["S3"],
      "explicit_citations": ["R1"],
      "inherited_citations": [],
      "references": ["R1"]
    }
  ]
}
```

Each `<pdf_stem>_extracted.json` file is now a single object with two
top-level lists: `references` (the bibliography, as before) and `claims`
(atomic claims pulled from the body text). `claims` is produced in two
stages — see `src/extract_references/claims.py` and
`src/extract_references/claim_extraction.py`:

1. **Deterministic (regex, no LLM)** — in-text citation markers, numbered
   (`[1]`, `[2, 3]`, `[4-6]`) or author-year (`(Smith et al., 2020)`,
   `Smith (2020)`), are resolved against the reference list and rewritten
   as canonical `<CIT:ref_id>` tags.
2. **LLM, per paragraph** — the model reads each tagged paragraph and
   extracts atomic claims, distinguishing `explicit_citations` (tagged in
   the claim's own sentence) from `inherited_citations` (only implied via
   discourse — a pronoun, "this approach", etc. — pointing back to an
   earlier sentence in the same paragraph).

Each reference's `claim_ids` lists which claims cite it. Pass `--no_claims`
to skip stage 2 and get references only (no extra LLM calls).

---

## Default MinerU command

```
.venv/bin/mineru -p {pdf} -o {out_dir} -b pipeline -m txt -d cpu -f false -t false
```

Override with `--mineru_cmd`. The template must contain `{pdf}` and `{out_dir}` placeholders.