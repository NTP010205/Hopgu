# Hợp Gu V2

Hệ gợi ý sản phẩm thương mại điện tử đa phương thức bằng **Streamlit + Python 3.11**. Ứng dụng tìm sản phẩm bằng tiếng Việt, ảnh tham chiếu hoặc cả hai; toàn bộ catalog/ảnh/review được lưu local để demo không phụ thuộc Internet sau bước chuẩn bị.

## Trạng thái artifact hiện tại

- 860 sản phẩm Tiki thật, 8 nhóm danh mục, giá VND.
- 14.219 review tiếng Việt liên kết bằng `product_id`.
- 860 ảnh local đã qua Pillow verify/convert RGB.
- Text embedding: `intfloat/multilingual-e5-small`, 384 chiều.
- Image embedding: `openai/clip-vit-base-patch32`, 512 chiều.
- Alpha fusion hiện tại: `0.8`, chọn trên validation set riêng.
- Ollama là tùy chọn runtime. Nếu không có, app chạy parser rule-based và hiển thị `FALLBACK` rõ ràng.

Số liệu trên được validator/smoke test kiểm tra từ artifact, không phải dữ liệu hiển thị mẫu.

## Chạy một lần trên Windows

```bat
run_project.bat
```

Script sẽ kiểm tra Python 3.11, tạo `.venv`, cài dependency nếu thiếu, kiểm tra dữ liệu/embedding rồi chạy một process Streamlit tại `http://localhost:8501`. Nếu máy chưa có Python 3.11 nhưng có `uv`, script tải Python managed vào `.python/` trong project.

Kiểm tra setup mà không mở server:

```bat
run_project.bat --check-only
```

Linux/macOS:

```bash
chmod +x run_project.sh
./run_project.sh
```

## Giao diện editorial kết nối AI thật — integration 2026-10-25

Kiến trúc: **HTML/CSS/JavaScript frontend → Python API adapter (`serve_editorial.py`) →
`RecommendationService` → E5 + CLIP + Qwen/Ollama**.

Backend source of truth là project `hopgu_v2` gốc, chỉ đọc khi tích hợp. Bản backend trong
project này có checksum trùng bản chuẩn; không viết lại parser, ranking hay fusion.
JavaScript chỉ gửi input và render kết quả, không thực hiện recommendation.

Từ thư mục chứa `serve_editorial.py`, chạy:

```bat
run_editorial.bat --check-only
run_editorial.bat
```

Mở `http://127.0.0.1:8000`. Script dùng Python 3.11 trong `.venv`, mặc định
`HOPGU_OFFLINE=1`. Model cache E5/CLIP phải có trong `data/models/`.
Giữ `HOPGU_FUSION_ALPHA` không đặt (mặc định **0.8**); không thay tên model.

Để dùng Qwen, chạy `ollama serve` nếu dịch vụ chưa chạy và bảo đảm đã cài
`qwen2.5:3b-instruct`. Nếu Ollama không kết nối được, backend dùng parser/rationale
fallback nguyên bản và UI hiển thị **FALLBACK**. Model E5/CLIP vẫn chạy thật.
`/api/status` nạp model và kiểm tra cache trước khi báo READY; lần khởi động đầu có thể chậm.

Các endpoint:

- `GET /api/status`: trạng thái runtime, model, artifact, alpha và Ollama.
- `GET /api/catalog`: catalog thật; hỗ trợ `keyword`, `category`, `min_price`,
  `max_price`, `min_rating`, `sort` (`reviews_desc`, `rating_desc`, `price_asc`, `price_desc`).
- `POST /api/recommend`: multipart gồm `query`, `image`, `category`, `min_price`,
  `max_price`, `top_k` (1–40). Query → `search_text`; ảnh → `search_image`;
  cả hai → `search_multimodal`. Không truyền alpha vào API.
- `GET /data/images/<filename>`: ảnh local, kiểm tra containment và MIME.

Upload tối đa 5 MB, JPG/PNG/WebP, được Pillow validate bằng hàm vision có sẵn.
Response giữ thứ tự ranking và score từ backend, số không có dữ liệu là `null`;
reason và review evidence lấy từ backend/catalog. Bộ lọc category/price không xử lý
bằng heuristic trong JavaScript. HTTP lỗi không có fallback sang mock catalog.

Trang Streamlit `05 Lookbook` nhúng URL server cùng origin với API: cần chạy
`run_editorial.bat` trước, rồi `run_project.bat`. Các file `hopgu_standalone_full.html`
và HTML tên cũ chỉ dẫn mở server; chúng không thể chạy AI bằng cách nhấp đúp offline.
Server vẫn chạy offline đối với inference/catalog; CSS CDN, font và ảnh người mẫu
editorial là tài nguyên trình bày từ Internet. Ảnh người mẫu không phải bằng chứng
sản phẩm xuất hiện trong ảnh. Card sản phẩm dùng ảnh local thật, thiếu ảnh hiện trạng thái thiếu.
Danh sách lưu chưa phải hệ thống đặt hàng; không có API checkout.

Kiểm chứng:

```bat
.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider
.venv\Scripts\python.exe -B scripts\smoke_test.py
.venv\Scripts\python.exe -B scripts\verify_editorial_api.py --expect READY
```

Khi Ollama thực sự không chạy, chạy verifier với `--expect FALLBACK`.
Verifier gửi multipart qua HTTP, gọi model thật, so sánh toàn bộ response với
kết quả gọi trực tiếp backend và lưu structured outputs tại `integration_verification/`.
Không chạy evaluate/tuning/rebuild trong integration này.
Xem `INTEGRATION_VALIDATION.md` để biết kết quả đã chạy và phần còn chưa kiểm chứng.
`integration_backup_20261025/` lưu bản trước khi sửa, chỉ là backup; server không phục vụ thư mục này.
`.venv_imported_huy_20261025/` giữ môi trường nhập từ máy cũ, không được dùng để chạy app.

Phân biệt trạng thái dịch vụ và trạng thái request: `/api/status` có thể READY vì Ollama
đang kết nối được, nhưng một request vẫn FALLBACK nếu parser thất bại hoặc backend
báo `llm_reason_error`. Adapter hiển thị cảnh báo explanation và giữ nguyên reason
dự phòng từ backend; không tự viết lại hay sửa guardrail để ép Qwen thành công.

## Chạy offline

Artifact model, ảnh và embedding phải được chuẩn bị trước. Khi chạy demo, script đặt `HOPGU_OFFLINE=1`; model chỉ được nạp từ `data/models/`, không tải mạng và không crawl Tiki.

Nếu chuyển project sang máy khác, giữ nguyên các thư mục:

- `data/processed/`
- `data/images/`
- `data/embeddings/`
- `data/models/`
- `data/eval/results/`

Manifest cache chỉ chứa `product_id`, checksum nội dung và đường dẫn tương đối; không fingerprint absolute path của máy.

## Chuẩn bị lại dữ liệu từ snapshot raw

Các lệnh này cần mạng ở bước tải ảnh/model, nhưng không chạy trong ứng dụng Streamlit:

```bat
.venv\Scripts\python.exe scripts\prepare_dataset.py --limit 900 --min-category-size 50
.venv\Scripts\python.exe scripts\download_images.py --target-count 800
.venv\Scripts\python.exe scripts\build_text_embeddings.py --batch-size 32
.venv\Scripts\python.exe scripts\build_image_embeddings.py --batch-size 16
.venv\Scripts\python.exe scripts\evaluate.py
```

Nguồn snapshot: repository công khai [phamkhoa18/tiki-data-analysis](https://github.com/phamkhoa18/tiki-data-analysis), commit `b95f3a667c5ff6c3e2e54143bb2980d2cdea0de2`. URL, ngày lấy và SHA-256 từng CSV nằm trong `data/raw/SOURCE.json`. Dữ liệu raw gồm bảng sản phẩm/review Tiki công khai phục vụ bài tập học thuật; trước khi tái phân phối cần tự kiểm tra điều khoản nguồn.

## Ollama / LLM

Mặc định cấu hình `qwen2.5:3b-instruct` tại `http://127.0.0.1:11434`:

```bash
ollama pull qwen2.5:3b-instruct
ollama serve
```

LLM được dùng để:

1. phân tích query thành JSON có cấu trúc;
2. viết lý do từ evidence đã truy xuất (tên, danh mục, giá, rating, review thật).

Prompt bắt buộc chỉ dùng evidence; LLM không tạo sản phẩm/review/giá. Nếu server hoặc model không sẵn sàng, `System Status` báo lỗi cụ thể và parser rule-based tiếp quản.

Có thể sao chép `.env.example` để tham khảo biến môi trường; app đọc trực tiếp biến môi trường của process.

## Test và kiểm chứng

```bat
set HOPGU_OFFLINE=1
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests
.venv\Scripts\python.exe scripts\smoke_test.py
```

Chạy Streamlit health check thủ công:

```bat
.venv\Scripts\python.exe -m streamlit run app.py --server.headless true
```

Sau đó mở `http://127.0.0.1:8501/_stcore/health`; response phải là `ok`.

`scripts/evaluate.py` đo Precision@5, Hit@5, nDCG@5 và latency cho text-only/image-only/multimodal. Alpha được chọn bằng 8 query validation; 24 query test không tham gia tuning. `data/eval/results/latest.json` có timestamp, model, checksum dataset và provenance; trang Đánh giá không hiển thị metric nếu file này thiếu hoặc không hợp lệ.

## Cấu trúc chính

```text
app.py                     Trang chủ Streamlit
pages/                     Tìm kiếm, Catalog, Đánh giá, System Status
src/data/                  Schema, loader, validator
src/nlp/                   E5, parser tiếng Việt, Ollama, review evidence
src/vision/                CLIP và validation upload
src/recommender/           Filter, score, fusion, service
src/evaluation/            Metric và provenance validation
scripts/                   Data/model/evaluation/smoke pipeline
data/                      Raw snapshot, ảnh, parquet, embeddings, eval
tests/                     Unit tests theo từng phase
```

## Giới hạn đã biết

- Snapshot hiện không có mọi ngành hàng Tiki; các query ngoài coverage vẫn được encode nhưng chất lượng phụ thuộc catalog.
- Metric manual dùng broad category làm relevance cùng với product ID đã gán; vì vậy đây là đánh giá retrieval theo nhóm sản phẩm, không phải log click/conversion người dùng.
- Ollama cần được cài/chạy riêng. Không có Ollama không làm app crash, nhưng đó là chế độ fallback chứ không phải LLM-ready.

## Lưu ý cho bản ZIP nộp E-learning

Để giảm dung lượng, bản ZIP nộp bài không kèm thư mục `data/models/`.

Source code, dữ liệu processed, ảnh sản phẩm, embeddings và kết quả đánh giá
vẫn được giữ nguyên.

Nếu chỉ xem hoặc chỉnh sửa source code thì không cần tải model.

Nếu muốn chạy đầy đủ ứng dụng trên máy mới, cần chuẩn bị lại hai model:

- `intfloat/multilingual-e5-small`
- `openai/clip-vit-base-patch32`

Có thể lấy lại thư mục `data/models/` từ bản full của project.

Nếu không có bản full, kết nối Internet và chạy pipeline chuẩn bị model:

```bat
run_project.bat --check-only

set HOPGU_OFFLINE=0
.venv\Scripts\python.exe scripts\build_text_embeddings.py --batch-size 32
.venv\Scripts\python.exe scripts\build_image_embeddings.py --batch-size 16
