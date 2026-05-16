# Usage Flow — Scientific Reference Extractor

> Tài liệu này mô tả chi tiết toàn bộ luồng xử lý của ứng dụng:  
> từ PDF đầu vào → qua MinerU → các bước heuristic → LLM → JSON đầu ra.

---

## Tổng quan kiến trúc

```
PDF
 │
 ▼
[__main__.py] ──builds──► ExtractionPipeline
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
          AsyncMinerUClient          AsyncLLMClient
          (content extraction)       (structured parsing)
                    │
                    ▼
             CitationParserEngine
             (heuristics / validation)
```

---

## Giai đoạn 0 — Entry Point (`__main__.py`)

**Trigger:** `python3 -m src.extract_references <pdf_path> [options]`

**Việc làm:**
1. Parse CLI arguments:
   - `pdf_path` – đường dẫn PDF đầu vào
   - `--llm_backend` – `openai` (default) | `ollama` | `together`
   - `--model` – tên model (default: `gpt-4o-mini`)
   - `--concurrency` – số tác vụ LLM đồng thời (default: 10)
   - `--output_dir` – thư mục lưu kết quả (default: `tests/json/minerU/`)
2. Dựng `AsyncLLMClient` với API key / base_url tương ứng:
   - `openai`: `OPENAI_API_KEY`, Instructor mode = **tool_call**
   - `together`: `TOGETHER_API_KEY`, base_url = together.xyz, Instructor mode = **JSON**
   - `ollama`: base_url = `localhost:11434/v1`, Instructor mode = **JSON**
3. Dựng `ExtractionPipeline` và gọi `asyncio.run(pipeline.run(pdf_path))`.
4. Serialize kết quả → `<output_dir>/<pdf_stem>_extracted.json`.

**Output của giai đoạn này:** `List[ExtractedCitation]` được dump ra file JSON.

---

## Giai đoạn 1 — MinerU: Chuyển PDF sang nội dung có cấu trúc

**File:** `clients.py` → `AsyncMinerUClient.extract_ref_blocks_with_page_idx()`  
**Retry:** tối đa 2 lần (tenacity), backoff 1–5 giây.

### Bước 1a — Chạy CLI MinerU
```
mineru -p <pdf> -o <out_dir> -b pipeline -m txt -d cpu -f false -t false
```
- Timeout: **300 giây**.
- Nếu exit code ≠ 0 → raise `RuntimeError`.

### Bước 1b — Ưu tiên `content_list.json` (primary path)
- Tìm `*_content_list.json` trong output dir, lấy file **lớn nhất**.
- Gọi `_extract_ref_blocks_with_pages(content_list)`.

### Bước 1c — Fallback: Markdown
- Nếu không có `content_list.json`, tìm `*.md` lớn nhất.
- Gọi `extract_references_from_markdown(markdown_text)`.
- Tất cả block trả về `page_idx = -1` (không có thông tin trang).

**Output của giai đoạn này:** `list[tuple[str, int]]` — danh sách `(ref_text, page_idx)`.

---

## Giai đoạn 2 — Phát hiện section References & Trích xuất block

**File:** `clients.py` → `AsyncMinerUClient._extract_ref_blocks_with_pages()`

### Các loại block MinerU nhận diện

| Nhóm | Các `type` | Xử lý |
|---|---|---|
| **Noise** (bỏ qua) | `page_number`, `header`, `page_footnote`, `footer`, `discarded` | Skip hoàn toàn |
| **Heading** | `title`, `section_header`, hoặc có `text_level` ≠ null | Dùng để xác định boundary |
| **Content** | `text`, `list`, `list_item`, `paragraph` | Thu thập vào buffer |

### Logic phát hiện section References
1. Duyệt từng item trong `content_list`.
2. Nếu item là **heading** và text (normalize lowercase) nằm trong:
   ```
   {"references", "reference", "bibliography", "works cited",
    "citations", "tai lieu tham khao", "tài liệu tham khảo"}
   ```
   → Bật cờ `in_references = True`, tiếp tục.
3. Khi đang `in_references`, gặp heading khác:
   - Nếu là major heading (`text_level <= 3`, độ dài < 120 ký tự) VÀ **không** trông như citation (không có năm, DOI, URL, "et al.", page range) → **dừng** thu thập.
4. Nếu text bắt đầu khớp `_POST_REF_STOP` (appendix, acknowledgments, supplement, conflict of interest, funding, notes, annex, about the author, ethical approval, declaration) → **dừng** thu thập.
5. Block `type: "list"` → extract từng `list_item` riêng lẻ.

---

## Giai đoạn 3 — Splitting: Tách block thành từng reference đơn lẻ

**File:** `clients.py` → `AsyncMinerUClient._split_content_list_refs()`

### Phân loại chế độ: Numbered vs Author-Year

Thuật toán đếm % dòng có **numbered marker** trong block:
```
[1]  hoặc  (1)  hoặc  1.  hoặc  1)
```
- Nếu **≥ 25%** dòng có marker → **Numbered mode**.
- Ngược lại → **Author-year mode**.

### Numbered mode
- Mỗi khi gặp dòng có marker → `flush()` buffer cũ, bắt đầu buffer mới.
- Marker bị **strip khỏi text** cuối cùng.
- Dòng tiếp theo không có marker → nối vào buffer hiện tại (continuation line).

### Author-year mode
- **Blank line** → hard boundary, `flush()` ngay.
- Dòng có chữ cái/số nhưng không blank → kiểm tra `_is_new_ref_start()`:

#### `_is_new_ref_start()` — Multi-signal scoring (ngưỡng >= 2 điểm)

| Signal | Điểm | Điều kiện |
|---|---|---|
| Dòng **trước** kết thúc bằng năm `(19\|20)XX[).,]` | +2 | Strong |
| Dòng **trước** kết thúc bằng DOI / URL | +2 | Strong |
| Dòng **hiện tại** là numbered ref `1. Title...` | +2 | Strong |
| Dòng **trước** kết thúc bằng page range `12-34` | +1 | Medium |
| Dòng **trước** kết thúc bằng volume pattern `12(3)` | +1 | Medium |
| Dòng **hiện tại** bắt đầu như Author surname | +1 | Medium |

Yêu cầu **tổng ≥ 2 điểm** → mới cắt ra entry mới (tránh cắt oan continuation line).

### Sau khi split
- Mỗi entry được `_normalize_reference_text()` (chuẩn hóa whitespace).
- Lọc bỏ entry có độ dài < 15 ký tự.

**Output của giai đoạn này:** `list[tuple[str, int]]` — các `(ref_text, page_idx)` riêng lẻ.

---

## Giai đoạn 4 — LLM Call #1: Resolve page-split (nếu có)

**File:** `pipeline.py` → `ExtractionPipeline._resolve_page_splits()`  
**LLM Method:** `AsyncLLMClient.decide_merge(text_a, text_b)`

### Khi nào được gọi?
- Khi có 2 block liền kề ở các **trang khác nhau** (`page_idx` khác nhau, cả hai ≥ 0).

### Fast-reject trước khi gọi LLM
- `_should_skip_merge(text_a, text_b)`: nếu `text_b` bắt đầu bằng numbered marker (`[1]`, `(1)`, `1.`) → **bỏ qua**, không cần hỏi LLM (chắc chắn là 2 ref riêng).

### LLM Call: `decide_merge`
```
Model: <configured model>
Retry: 3 lần, backoff 2–10 giây
```

**System prompt:**
> "You are a bibliographic citation expert. Two text fragments were extracted from adjacent pages of a PDF. Determine whether they are TWO SEPARATE references or ONE reference split across pages. Answer should_merge=true ONLY if fragment B is clearly a continuation of fragment A (same bibliographic entry). Answer should_merge=false if fragment B starts a new, independent reference."

**User prompt:**
```
Fragment A (end of page):
<text_a>

Fragment B (start of next page):
<text_b>
```

**Schema trả về:** `MergeDecision`
```python
class MergeDecision(BaseModel):
    should_merge: bool   # True = ghép lại, False = giữ riêng
    reason: str          # Giải thích ngắn gọn
```

**Xử lý kết quả:**
- Tất cả boundary pair được kiểm tra **song song** (`asyncio.gather`).
- Nếu `should_merge=True` → nối `text_a + " " + text_b` thành 1 entry.
- Nếu exception → giữ riêng (fallback safe).

**Output của giai đoạn này:** `List[str]` — raw citation strings đã được merge nếu cần.

---

## Giai đoạn 5 — Pre-processing heuristic trước LLM

**File:** `pipeline.py` → `ExtractionPipeline._process_single_citation()`  
**Heuristic:** `CitationParserEngine.heal_broken_urls(raw_text)`

Áp dụng trước khi gửi lên LLM:
1. Fix URL bị phân mảnh do line break PDF:
   - `https : //` → `https://`
   - `DOI : 10 .` → `DOI:10.`
2. Compact URL block (xóa spaces nội tại trong URL/DOI string).

---

## Giai đoạn 6 — LLM Call #2 (hoặc #3): Extract citation metadata

**File:** `pipeline.py` → `ExtractionPipeline._process_single_citation()`  
Tất cả citation được xử lý **song song** (semaphore = `max_concurrency`).

### Nhánh A — Standard (raw_text ≤ 1000 ký tự)

**LLM Method:** `AsyncLLMClient.extract_citation(raw_text)`
```
Model: <configured model>
Retry: 3 lần, backoff 2–10 giây
```

**System prompt:**
> "You are a high-precision citation extraction specialist. Extract metadata from the provided raw text. If information like DOI or arXiv ID is not present, leave it null. Do not use external knowledge or invent data."

**User prompt:**
```
raw_text: <raw_text>
```

**Schema trả về:** `ParsedCitationEntry`
```python
class ParsedCitationEntry(BaseModel):
    title: str                   # Bắt buộc
    authors: List[str]           # Dạng "Given Family"
    venue: Optional[str]         # Tên journal / conference / repository
    year: Optional[int]          # 4 chữ số
    doi: Optional[str]
    arxiv_id: Optional[str]
    url: Optional[str]
```

### Nhánh B — Batch (raw_text > 1000 ký tự → nghi là nhiều ref ghép)

**LLM Method:** `AsyncLLMClient.extract_citations_batch(raw_text)`
```
Model: <configured model>
Retry: 2 lần, backoff 2–10 giây
```

**System prompt:**
> "Extract each individual bibliographic citation from the text into a structured collection."

**User prompt:** `<raw_text>` (toàn bộ)

**Schema trả về:** `CitationCollection`
```python
class CitationCollection(BaseModel):
    citations: List[ParsedCitationEntry]
```

**Nếu batch fail** → fallback sang nhánh A (standard single extraction).

---

## Giai đoạn 7 — Validation & Post-processing

**File:** `pipeline.py` → `_process_single_citation()` + `_post_process_results()`

### `is_plausible_reference()` — Anti-hallucination filter

| Case | Điều kiện | Kết quả |
|---|---|---|
| raw_text rỗng hoặc < 10 ký tự | — | Loại bỏ |
| **Case 1**: Có `title` | `fields_present >= 2` (trong: authors, title, venue, year, identifier) | Giữ lại |
| **Case 2**: Không có `title` | Có strong identifier (doi / arxiv_id / url) **VÀ** (authors hoặc year) | Giữ lại |
| Còn lại | — | Loại bỏ |

### Re-indexing
- Tất cả citation được flatten và đánh lại index tuần tự: `R1`, `R2`, `R3`, ...
- Batch items trước khi merge có dạng `R<idx>_<sub>` (ví dụ: `R3_1`, `R3_2`).

---

## Giai đoạn 8 — Output

**File:** `__main__.py`

Mỗi `ExtractedCitation` được serialize với `model_dump(exclude_none=True)`:

```json
[
  {
    "ref_id": "R1",
    "raw_text": "LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521(7553), 436-444.",
    "title": "Deep learning",
    "authors": ["Yann LeCun", "Yoshua Bengio", "Geoffrey Hinton"],
    "venue": "Nature",
    "year": 2015,
    "identifiers": {
      "doi": "10.1038/nature14539"
    }
  },
  ...
]
```

**Lưu tại:** `<output_dir>/<pdf_stem>_extracted.json`

---

## Sơ đồ luồng tổng hợp

```
PDF
 │
 ▼ [Stage 0] __main__.py
 │  Parse args → Build LLMClient → Build Pipeline
 │
 ▼ [Stage 1] AsyncMinerUClient.extract_ref_blocks_with_page_idx()
 │  Run MinerU CLI (timeout 300s)
 │  → Primary: content_list.json
 │  → Fallback: *.md
 │  Output: list[(ref_text, page_idx)]
 │
 ▼ [Stage 2] _extract_ref_blocks_with_pages()
 │  Scan content_list → detect "References" heading
 │  Collect text/list blocks → stop at post-ref sections
 │  Output: raw blocks with page_idx
 │
 ▼ [Stage 3] _split_content_list_refs()
 │  Classify: Numbered (≥25% markers) vs Author-year
 │  Split into individual reference strings
 │  Output: list[(ref_text, page_idx)]
 │
 ▼ [Stage 4] _resolve_page_splits()  ← LLM CALL #1: decide_merge
 │  Detect cross-page boundaries
 │  Fast-reject numbered markers
 │  → LLM: MergeDecision {should_merge, reason}
 │  Merge if needed
 │  Output: List[str] (raw citation strings)
 │
 ▼ [Stage 5] heal_broken_urls()
 │  Fix URL/DOI fragmentation (deterministic)
 │
 ▼ [Stage 6] _process_single_citation()  ← LLM CALL #2 or #3
 │  If len > 1000: extract_citations_batch() → CitationCollection
 │                                  ↑ LLM CALL #2 (batch)
 │  Else:          extract_citation()       → ParsedCitationEntry
 │                                  ↑ LLM CALL #3 (single)
 │
 ▼ [Stage 7] is_plausible_reference() + _post_process_results()
 │  Validate, filter, re-index (R1, R2, ...)
 │
 ▼ [Stage 8] JSON Output
    <pdf_stem>_extracted.json
```

---

## Tóm tắt các LLM Call

| # | Method | Schema trả về | Retry | Khi nào gọi |
|---|---|---|---|---|
| **LLM #1** | `decide_merge(text_a, text_b)` | `MergeDecision` | 3x | Mỗi cặp block liền kề khác trang (song song) |
| **LLM #2** | `extract_citations_batch(raw_text)` | `CitationCollection` | 2x | Khi raw_text > 1000 ký tự |
| **LLM #3** | `extract_citation(raw_text)` | `ParsedCitationEntry` | 3x | Khi raw_text ≤ 1000 ký tự (hoặc batch fail) |

Tất cả LLM call đều dùng **Instructor** (Pydantic-enforced structured output).  
Concurrency được kiểm soát bởi `asyncio.Semaphore(max_concurrency)`.

---

## Các quy tắc trích xuất reference (Rules)

### R1 — Phát hiện heading References
- Normalize lowercase, strip `#`, so sánh exact với whitelist.
- Hỗ trợ cả tiếng Anh và tiếng Việt.

### R2 — Dừng thu thập
- Gặp major heading (`text_level <= 3`, < 120 ký tự) không trông như citation.
- Gặp text match `_POST_REF_STOP` (appendix, acknowledgments, v.v.).

### R3 — Noise filtering
- Loại bỏ hoàn toàn: `page_number`, `header`, `footer`, `page_footnote`, `discarded`.

### R4 — Splitting mode classification
- ≥ 25% dòng có marker → Numbered mode; ngược lại → Author-year mode.

### R5 — Continuation line (Author-year)
- Không cắt nếu tổng score < 2 (tránh cắt giữa chừng 1 reference).

### R6 — Page-split resolution
- Fast-reject nếu fragment B bắt đầu bằng numbered marker.
- Nếu không, hỏi LLM (song song cho tất cả boundary pairs).

### R7 — URL healing (deterministic, trước LLM)
- Fix `https : //` → `https://`, DOI whitespace, URL internal spaces.

### R8 — Batch threshold
- Nếu raw_text > 1000 ký tự → dùng batch LLM call (extract nhiều citation 1 lần).

### R9 — Plausibility filter (sau LLM)
- Có title: cần ≥ 2 fields (authors / title / venue / year / identifier).
- Không có title: cần strong identifier + (authors hoặc year).
- raw_text < 10 ký tự → luôn loại bỏ.

### R10 — Anti-hallucination
- LLM được yêu cầu không dùng kiến thức bên ngoài, không bịa dữ liệu.
- Instructor đảm bảo output đúng schema (không phải free-text).

### R11 — Re-indexing
- Sau khi flatten toàn bộ kết quả: đánh index tuần tự `R1`, `R2`, ...
- `null` fields được loại bỏ khỏi JSON output (`exclude_none=True`).
