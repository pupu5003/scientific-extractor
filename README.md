# Scientific Reference Extractor

A high-precision pipeline for extracting structured bibliographic references from scientific PDFs using **MinerU** and an **LLM** (via `instructor`).

---

## How it works

1. **MinerU** converts the PDF to structured JSON (`content_list.json`) or Markdown.
2. Reference blocks are located and split into individual entries using deterministic heuristics.
3. Pairs of entries at page boundaries are checked by the LLM to detect incorrect splits.
4. Each entry is parsed by the LLM into a structured schema (`ParsedCitationEntry`).
5. Non-plausible entries are filtered out and results are saved as JSON.

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
| `--concurrency` | `10` | Max citations parsed concurrently |
| `--output_dir` | `tests/json/minerU/` | Directory to save the output JSON |
| `--mineru_cmd` | *(see below)* | MinerU command template |

**Examples:**

```bash
# OpenAI (default)
python3 -m src.extract_references tests/paper.pdf

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
| `--citation_workers` | `10` | Max citations parsed concurrently per PDF |
| `--mineru_cmd` | *(see below)* | MinerU command template |
| `--debug_markdown_dir` | *(empty)* | If set, saves MinerU markdown output here for inspection |

**Examples:**

```bash
# Process all PDFs in a folder
python3 run_batch.py tests/pdfs/ --output_dir out/

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
[
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
    }
  }
]
```

---

## Default MinerU command

```
.venv/bin/mineru -p {pdf} -o {out_dir} -b pipeline -m txt -d cpu -f false -t false
```

Override with `--mineru_cmd`. The template must contain `{pdf}` and `{out_dir}` placeholders.