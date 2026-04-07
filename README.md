# Scientific Reference Extractor 🚀

A high-precision pipeline for extracting structured references from scientific PDFs using **MinerU** and **LLM (Instructor)**.

## 🌟 Key Features
- **Pure LLM Extraction**: Uses OpenAI (GPT-4o-mini) and `instructor` for schema-driven, zero-hallucination parsing.
- **Smart Reference Splitting**: Handles dense reference blocks without markers (e.g., ICLR/NeurIPS papers) via deterministic heuristics and LLM batch fallback.
- **MinerU Integration**: Uses `magic-pdf` (MinerU) for robust PDF-to-Structured-JSON conversion.
- **Noise Filtering**: Automatically skips headers, footers, and page numbers during extraction.

## 🛠️ Prerequisites
- Python 3.10+
- OpenAI API Key (set in `.env`)
- Bộ công cụ MinerU (`magic-pdf`, `mineru`)

## 🚀 Setup

1. **Clone & Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Environment Variables**:
   Create a `.env` file from `.env.example`:
   ```bash
   OPENAI_API_KEY=your_key_here
   LLM_BACKEND=openai
   LLM_MODEL=gpt-4o-mini
   ```

3. **MinerU Models (One-time)**:
   ```bash
   .venv/bin/mineru-models-download
   ```

## 📖 Usage

### 1. Extract from a single PDF
The pipeline runs MinerU automatically and then parses the references:
```bash
.venv/bin/python -m src.extract_references tests/2510.04871v1.pdf
```

### 2. Extract from a MinerU output folder
If you already ran MinerU, you can point the pipeline to the output folder to save time:
```bash
.venv/bin/python -m src.extract_references tests/mineru_output/10201/
```

### 3. Batch Mode
Process multiple PDFs in parallel:
```bash
.venv/bin/python run_batch.py "tests/pdfs/*.pdf" --output_dir tests/json/minerU/
```

## 📊 Output Format
The results are saved as structured JSON in `tests/json/minerU/`.

```json
{
  "ref_id": "R1",
  "title": "Example Paper Title",
  "authors": ["Author One", "Author Two"],
  "venue": "Nature Communications",
  "year": 2024,
  "identifiers": {
    "doi": "10.1038/...",
    "arxiv_id": "2401.xxxxx",
    "url": "https://..."
  }
}
```


.venv/bin/mineru -p tests/pdfs/minerU/09669_Progressive_Compositionality_in_Text-to-Image_Generative_Models.pdf -o /tmp/mineru_inspect -b pipeline -m txt -d cpu -f false -t false


.venv/bin/mineru -p tests/pdfs/minerU/10208_Adaptive_Gradient_Clipping_for_Robust_Federated_Learning.pdf -o tests/pdfs/minerU_results/ -b pipeline -m txt -d cpu -f false -t false