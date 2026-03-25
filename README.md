# Scientific Reference Extractor 📄🔍

An enterprise-grade, highly concurrent pipeline for extracting and structuring scientific citations from PDF manuscripts. This module serves as the **Extractor Agent** for the larger Scientific References Verification framework.

## Architecture
This pipeline utilizes a hybrid deterministic/probabilistic approach:
1. **GROBID (Deterministic)**: Extracts raw citation strings and basic TEI XML structure.
2. **Heuristic Engine**: Applies resilient regex fallbacks to catch missing DOIs and arXiv IDs.
3. **LLM Validator (Probabilistic)**: Acts as a Confidence Router. If deterministic parsing fails, an LLM (OpenAI or local Ollama) is invoked to patch missing fields using strict JSON schemas.

## Prerequisites
- Python 3.10+
- A running instance of [GROBID](https://grobid.readthedocs.io/en/latest/) (usually via Docker: `docker run -t --rm -p 8070:8070 lfoppiano/grobid:0.8.0`)
- An OpenAI API Key (or a local Ollama instance).

## Setup
1. Clone the repository and navigate to the directory.
2. Create a virtual environment: `python -m venv venv && source venv/bin/activate`
3. Install dependencies: `make install`
4. Copy `.env.example` to `.env` and fill in your variables.

## Usage
Run the pipeline via the CLI module:

```python -m src.extract_references path/to/your/paper.pdf --llm_backend openai```

## Output:
The pipeline will generate a strictly typed `paper.pdf_extracted.json` file conforming to the Verification Database schema.


## Run batch
``` python3 run_batch.py "tests/pdfs/iclr2025/spotlight/*.pdf" --llm_backend together --pdf_workers 5 --output_dir tests/json/iclr2025_anystyle/```