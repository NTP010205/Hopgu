# Validation log — Hợp Gu V2

Ngày kiểm chứng: 2026-10-11 (Asia/Saigon)

Tài liệu này chỉ ghi kết quả đã chạy thực tế. Mục chưa chạy hoặc thiếu dependency ngoài project được ghi rõ `BLOCKED`/`NOT VERIFIED`.

## Môi trường

- Python project-managed: `3.11.15` trong `.python/` + `.venv/`.
- Streamlit: `1.58.0`.
- Torch: `2.14.1` CPU.
- Text model local: `intfloat/multilingual-e5-small`.
- Vision model local: `openai/clip-vit-base-patch32`.
- Ollama executable/server: không phát hiện/không kết nối được.

## Phase 1 — skeleton, schema, sample thật, UI tối thiểu

Đã chạy:

```text
python scripts/prepare_dataset.py --limit 60
python scripts/download_images.py --target-count 50
python -m unittest tests.test_phase1 -v
python -c "from src.data.loader import load_products,load_reviews; ..."
Invoke-WebRequest http://127.0.0.1:8508/_stcore/health
```

Kết quả thực tế tại thời điểm Phase 1:

- 50 sản phẩm Tiki thật, 1.475 review thật, 8 danh mục.
- 5/5 unit test qua.
- Dataset validator qua.
- Streamlit health: HTTP 200, body `ok`.
- Validator đã bắt một lỗi nguồn có `product_id` trùng; pipeline được sửa để deduplicate trước khi phase được xác nhận.

## Phase 2 — full dataset, ảnh local, text embeddings

Đã chạy pipeline nhiều lần sau khi sửa taxonomy; artifact cuối là:

- `products.parquet`: 800 sản phẩm.
- `reviews.parquet`: 14.090 review có nội dung.
- 8 broad categories.
- 800 ảnh local đã qua Pillow verify/convert RGB.
- Text embedding shape: `[800, 384]`, `float32`, normalized.
- Text dataset SHA-256: `7d523eb7298af02ace6e154730306224f220cd9fe601f232d18346c657b052a3`.

Integration thật đã dùng E5 để chạy 5 query tiếng Việt bắt buộc. Truy vấn `tai nghe bluetooth chống ồn` trả tai nghe Sony WF-1000XM5 ở hạng 1. Sau warmup, smoke cuối đo text search `13,92 ms` trên máy này.

## Phase 3 — CLIP và image-only

Đã chạy:

```text
.venv\Scripts\python.exe scripts\build_image_embeddings.py --batch-size 16
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests\test_phase3.py
```

Kết quả:

- Image embedding shape: `[800, 512]`, `float32`, normalized.
- Image dataset SHA-256: `63c428519139c9d56fe06af0e16370f09f19ea2e5bfc738e4d1dc1fad6a41471`.
- 4 test upload validation qua (PNG/RGB, bytes lỗi, extension lỗi, giới hạn dung lượng).
- Real self-image query trả đúng `product_id` ở top 1 với cosine `1.000000`.
- Smoke cuối sau warmup: image search `34,59 ms`.

## Phase 4 — parser, LLM status, multimodal fusion

Đã chạy unit test parser cho 5 câu:

1. `túi đeo chéo nữ màu đen dưới 500 nghìn`
2. `giày chạy bộ nam nhẹ`
3. `tai nghe bluetooth chống ồn`
4. `kem chống nắng cho da dầu`
5. `balo laptop chống nước`

Kết quả:

- Parser fallback nhận đúng giá `500.000`, gender/category/attribute được assert trong test.
- Fusion chuẩn hóa riêng text/image trước khi áp alpha.
- Real multimodal query với ảnh Sony WF-1000XM5 trả đúng sản phẩm ở top 1.
- Smoke cuối sau warmup: multimodal `47,25 ms`.
- Ollama status thực tế: `ready=false`, `mode=fallback`, chi tiết `Không kết nối được Ollama: ConnectTimeout`.

`BLOCKED — OLLAMA REAL PATH`: code gọi `/api/tags`, `/api/chat`, JSON parser và evidence-grounded reason đã được triển khai nhưng chưa thể kiểm chứng inference LLM thật do máy không có Ollama/model `qwen2.5:3b-instruct`. Không ghi PASS cho nhánh này.

## Phase 5 — evaluation và alpha

Đã chạy:

```text
set HOPGU_OFFLINE=1
.venv\Scripts\python.exe scripts\evaluate.py
```

Query set gồm 32 query tiếng Việt: 8 validation + 24 test. Alpha được chọn trên validation, không dùng test. Alpha chọn được: `0.8`.

Manual test (24 query, broad-category relevance + product ID gán trong JSONL):

| Hệ | Precision@5 | Hit@5 | nDCG@5 | Latency trung bình |
|---|---:|---:|---:|---:|
| text-only | 0,8917 | 1,0000 | 0,9145 | 13,14 ms |
| image-only | 0,8333 | 1,0000 | 0,8692 | 33,20 ms |
| multimodal | 0,9250 | 1,0000 | 0,9433 | 42,76 ms |

Proxy test (32 query từ title/description/image, cùng broad category là relevant) được ghi riêng trong `latest.json`; không trình bày như human relevance.

Artifact:

- `data/eval/results/latest.json`
- `data/eval/results/latest_details.csv`

## Phase 6 — final QA

Lệnh cuối đã chạy:

```text
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests
.venv\Scripts\python.exe scripts\smoke_test.py
run_project.bat --check-only
uv pip check --python .venv\Scripts\python.exe
```

Kết quả:

- `19 passed, 5 subtests passed`.
- Smoke: `ok=true`, Python `3.11.15`, 800 products, 14.090 reviews, 8 categories.
- `run_project.bat --check-only`: `[OK] Python 3.11, dependencies, and required artifacts are ready.`
- `uv pip check`: 79 package, tất cả compatible.
- Streamlit AppTest: 5/5 file có `exceptions=0`.
- Chạy thật bằng `run_project.bat`: root HTTP 200; `/_stcore/health` HTTP 200, body `ok`; server dừng sạch sau kiểm tra.
- Chế độ `HOPGU_OFFLINE=1` đã dùng trong smoke và evaluation; E5/CLIP nạp từ local cache.

## Chưa kiểm chứng / blocker còn lại

- `BLOCKED`: Ollama + `qwen2.5:3b-instruct` thực tế, như mô tả ở Phase 4.
- `NOT VERIFIED`: chạy trên một máy vật lý sạch thứ hai. Đã tạo mới Python 3.11 managed và `.venv` trong workspace trên máy hiện tại, nhưng đó không thay thế kiểm thử trên máy khác.
- `NOT VERIFIED`: thực thi `run_project.sh`; môi trường không có Git Bash và WSL bị từ chối. File đã được review cú pháp thủ công, nhưng không ghi PASS.

## Phase 7 — LLM Verification & Final Hardening

Trạng thái: `BLOCKED — OLLAMA CHƯA KHẢ DỤNG`.

Khảo sát thực tế ngày 2026-10-11:

- `where.exe ollama`: không tìm thấy executable trong `PATH`.
- Không có `ollama.exe` tại các vị trí cài đặt Windows phổ biến đã kiểm tra.
- Không có process Ollama và không có listener trên cổng `11434`.
- `GET http://127.0.0.1:11434/api/tags`: connection refused.
- `OllamaService().status()`: `ready=false`, `mode=fallback`, chi tiết `Không kết nối được Ollama: ConnectTimeout`.

Vì blocker trên, chưa chạy inference thật với `qwen2.5:3b-instruct`; chưa kiểm chứng sáu truy vấn JSON, reason theo evidence, so sánh LLM/fallback, ba mode khi LLM bật, regression sau LLM và clean packaging. Không mục nào trong số này được ghi PASS.

Lệnh cần chạy trong PowerShell để gỡ blocker:

```powershell
winget install --id Ollama.Ollama -e
```

Mở PowerShell mới và giữ server chạy:

```powershell
ollama serve
```

Trong PowerShell thứ hai:

```powershell
ollama pull qwen2.5:3b-instruct
ollama list
curl.exe http://127.0.0.1:11434/api/tags
```

Chỉ tiếp tục Phase 7 khi `/api/tags` liệt kê đúng model `qwen2.5:3b-instruct`.

### Cập nhật sau khi gỡ blocker — 2026-10-11

Trạng thái cuối: `COMPLETE — VERIFIED ON CURRENT WINDOWS MACHINE`.

Blocker Ollama ở phần trên được giữ nguyên như lịch sử lần kiểm tra ban đầu. Sau khi cài/chạy Ollama, đã xác minh lại:

- Executable: `C:\Users\Binnho\AppData\Local\Programs\Ollama\ollama.exe`.
- Ollama version: `0.40.3`.
- API: `http://127.0.0.1:11434/api/tags`.
- Model thật: `qwen2.5:3b-instruct`, digest `357c53fb659c`, dung lượng hiển thị `1.9 GB`.
- Trạng thái cuối từ project: `ready=true`, `mode=ollama`, `Ollama sẵn sàng với qwen2.5:3b-instruct`.

Các lệnh/chương trình kiểm chứng chính đã chạy thật:

```text
ollama.exe --version
ollama.exe list
GET http://127.0.0.1:11434/api/tags
.venv\Scripts\python.exe scripts\verify_phase7_parser.py
.venv\Scripts\python.exe scripts\verify_phase7_integration.py
.venv\Scripts\python.exe scripts\verify_phase7_fallback.py
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests
.venv\Scripts\python.exe scripts\smoke_test.py
run_project.bat --check-only
Streamlit AppTest cho app.py và 4 file pages/*.py
```

#### LLM parser thật

Đã gọi Qwen thật cho đủ sáu truy vấn:

1. `túi đeo chéo nữ màu đen dưới 500 nghìn`
2. `giày chạy bộ nam nhẹ dưới 1 triệu`
3. `tai nghe bluetooth chống ồn`
4. `kem chống nắng cho da dầu`
5. `balo laptop chống nước`
6. `tôi muốn một chiếc túi giống hình này nhưng rẻ hơn 600 nghìn`

Raw JSON của LLM, output sau validation và output fallback đối chiếu được lưu tại `data/verification/phase7_parser_outputs.json`. Cả 6 output cuối có `parser_mode=ollama` và đủ các field parser bắt buộc.

Inference đầu tiên đã phát hiện lỗi thật: Qwen đặt sai cận giá; fallback chưa hiểu `rẻ hơn`; màu `hồng` bị match nhầm trong `chống`; màu `tím` bị match nhầm từ động từ `tìm`; LLM có thể tự gán category cho query generic. Parser/prompt/guard đã được sửa tối thiểu và thêm regression test. Lần chạy cuối xác nhận các cận `500.000`, `1.000.000`, `600.000` nằm ở `price_max`, không còn màu/category bịa cho query generic.

#### Evidence và explanation

Đã phát hiện `reviews_text` thật được load dưới dạng `numpy.ndarray`, trong khi code cũ chỉ nhận `list/tuple`, làm review evidence rỗng. `top_review_evidence()` đã được sửa để đọc iterable artifact thật.

Đã kiểm chứng 5 recommendation text-only và 5 recommendation multimodal. Với từng sản phẩm, artifact lưu nguyên evidence đầu vào và raw reason của LLM. Guard hiện yêu cầu:

- đúng `product_id` đã retrieve;
- score, giá và rating khớp evidence;
- trích dẫn review là substring nguyên văn;
- response sai ID, thiếu reason hoặc diễn giải ngoài evidence bị từ chối.

Kết quả cuối: 10/10 reason qua guard; không có numeric claim ngoài evidence. Chi tiết tại `data/verification/phase7_integration.json`.

#### Ba mode và multimodal

- Text-only: 5 kết quả thật; E5 chạy; parser/reason dùng Ollama.
- Image-only: dùng ảnh thật `data/images/3304875.jpg`; đúng sản phẩm `3304875` ở top 1, cosine `1.0000001192`.
- Text + image: query `tìm sản phẩm giống hình nhưng dưới 1 triệu`; E5, CLIP, fusion và LLM explanation đều chạy.
- Multimodal output có `text_score`, `image_score`, `final_score`; cả 5 kết quả đều `price_vnd <= 1.000.000`.
- Alpha vẫn là `0.8`.

Latency của lượt integration cuối trên workspace hiện tại: text-only `16.719,575 ms`, image-only `804,146 ms`, multimodal `4.816,534 ms`. Smoke sau đó cảnh báo text/multimodal warm latency vượt mục tiêu cũ do reason được gọi riêng từng sản phẩm để guard evidence; không che warning này.

#### READY → FALLBACK → READY

Đã tắt đúng hai process Ollama và xác nhận API cổng `11434` không phản hồi. Khi OFF:

- `OllamaService`: `ready=false`, `mode=fallback`, `ConnectTimeout`;
- parser rule-based chạy;
- text search trả 3 kết quả thật, đều dưới 1 triệu;
- System Status AppTest có `exceptions=0` và hiển thị cảnh báo `FALLBACK`.

Artifact: `data/verification/phase7_fallback.json`.

Sau đó đã chạy lại `ollama.exe serve`; `/api/tags`, project client và System Status AppTest đều xác nhận `READY` với `qwen2.5:3b-instruct`.

#### Regression

- Pytest: `27 passed, 5 subtests passed`.
- Smoke: `ok=true`, LLM `ready=true`, parser `ollama`; text warm `5.205,98 ms`, image warm `126,84 ms`, multimodal warm `5.288,42 ms`. Hai cảnh báo latency được giữ nguyên.
- `run_project.bat --check-only`: PASS, thông báo `[OK] Python 3.11, dependencies, and required artifacts are ready.`
- Streamlit AppTest: `5/5` file có `exceptions=0`.

#### Clean-copy test

Đã tạo bản copy tại `D:\Thực hành\TTNTUD\project\hopgu_v2_clean_phase7`. Trước bootstrap, bản copy không chứa `.venv`, `.python`, `.cache`, `__pycache__`, `.pytest_cache`, `.pyc`/`.pyo`; vẫn có `data/processed`, `data/images`, `data/embeddings`, `data/models`, `data/eval/results` và `data/verification`.

Kết quả chạy trong clean copy:

- `run_project.bat --check-only` tự cài Python `3.11.15`, tạo `.venv` mới, cài 79 package và PASS.
- `run_project.bat` chạy server thật; root HTTP `200`, `/_stcore/health` HTTP `200`, body `ok`; server sau đó đã dừng.
- Script integration trong clean copy PASS text-only, image-only, multimodal, price filter, fusion, LLM explanation và alpha `0.8`.
- AppTest clean copy: `app.py` và 4 trang đều `exceptions=0`; Catalog, Evaluation và System Status load thành công.

#### Metric và blocker còn lại

- Không sửa evaluation dataset, label, metric hoặc alpha; không chạy lại evaluation.
- `data/eval/results/latest.json` vẫn có `generated_at=2026-10-11T06:01:27.042692+00:00`, alpha `0.8`, SHA-256 file `D88F960E2771E9E4B8FA5F72945C5604D8747B4890BB73E59377C2E8F38AF8F4`.
- `NOT VERIFIED`: máy vật lý thứ hai.
- `NOT VERIFIED`: `run_project.sh`, như lịch sử Phase 6.

## Phase 8 — Relevance + UI + Catalog

Ngày kiểm chứng: 2026-10-12 (Asia/Saigon).

Trạng thái: `COMPLETE — VERIFIED ON CURRENT WINDOWS MACHINE`.

Lịch sử `BLOCKED` của Ollama ở Phase 4/7 phía trên được giữ nguyên. Blocker đó
đã được gỡ từ Phase 7 và lần kiểm tra cuối Phase 8 vẫn xác nhận:
`ready=true`, `mode=ollama`, model `qwen2.5:3b-instruct`.

### Baseline trước khi sửa code

Đã chạy 5 ca image-only và 5 ca text + image bằng ảnh thật trong catalog:

- túi: catalog không có sản phẩm tên “túi xách” hoặc “balo”, nên dùng ảnh thật gần
  nhất là bộ hộp cơm có túi mang theo; nhóm thay thế thứ năm là sách;
- giày: `277426915`;
- tai nghe: `272030678`;
- mỹ phẩm: `55613571`;
- sách: `3304875`.

Artifact `data/verification/phase8_baseline.json` lưu parsed category, source
category, từng `text_score`, `image_score`, `final_score` và phân bố danh mục
top-20. Kết quả baseline trên đủ 24 query test có ảnh:

| Hệ | Category Precision@20 trước |
|---|---:|
| image-only | 0,7271 |
| multimodal | 0,8604 |

Nguyên nhân gốc quan sát được:

- image-only xếp hạng toàn bộ catalog, nên top-20 lẫn nhiều broad category dù ảnh
  tự truy vấn thường đứng đầu;
- parser chưa ánh xạ “túi xách” và “giày” vào taxonomy hiện có;
- khi text không tạo được category, multimodal dùng candidate toàn catalog và
  query chung chung có thể chi phối fusion. Alpha vẫn đúng `0.8`; không retune.

### Thay đổi relevance

- Bổ sung CLIP zero-shot broad-category classifier bằng prompt ensemble trong cùng
  không gian vector CLIP.
- Confidence cao dùng hard constraint một danh mục; confidence trung bình mở rộng
  top-2; confidence thấp tìm toàn catalog và metadata/UI ghi rõ “chưa chắc chắn”.
- Multimodal ưu tiên category rõ từ text hoặc bộ lọc người dùng trước category ảnh.
- Bộ lọc giá được áp dụng trước scoring và là ràng buộc nghiêm ngặt.
- Pipeline cuối: parse → price/category candidates → text/image scores → normalize
  riêng → fusion `alpha=0.8` → top-K.
- Default top-K là 20; UI chỉ cho chọn 10/20/40. Metric cũ vẫn đánh giá @5.

Kiểm tra zero-shot riêng trên 24 ảnh test: top-1 category đúng 87,5%; category thật
nằm trong candidate pool ở 91,7%. Đây là kiểm tra sau khi chọn ngưỡng guardrail cố
định, không dùng để retune alpha.

Artifact `data/verification/phase8_relevance_after.json` lưu đầy đủ 5 ca sau sửa.
Parser Ollama thật trả:

| Ca | Parsed category | Image-only top-20 | Multimodal top-20 | Giá |
|---|---|---|---|---|
| túi xách | Thời trang nữ | 20 Nhà cửa & Đời sống | 20 Thời trang nữ | đúng |
| giày | Thời trang nam | 20 Thời trang nam | 20 Thời trang nam | đúng |
| tai nghe | Điện tử & Phụ kiện | 20 Điện tử & Phụ kiện | 20 Điện tử & Phụ kiện | đúng |
| kem chống nắng | Làm đẹp & Sức khỏe | 20 Làm đẹp & Sức khỏe | 20 Làm đẹp & Sức khỏe | đúng |
| sách | Sách | 20 Sách | 20 Sách | đúng |

Ca “túi xách” cố ý thể hiện quy tắc ưu tiên text: ảnh nguồn thực tế là túi đựng bộ
hộp cơm thuộc Nhà cửa & Đời sống, nhưng query nói rõ “túi xách”, nên multimodal
giới hạn Thời trang nữ.

### Evaluation sau thay đổi

Trước khi chạy lại evaluation, kết quả Phase 7 đã được lưu tại:

- `data/eval/results/archive/phase7_latest_20261011T060127Z.json`
- `data/eval/results/archive/phase7_latest_details_20261011T060127Z.csv`

`scripts/evaluate.py` được chạy với alpha cố định `0.8`;
`alpha_selection.selected_on=fixed_from_previous_phase_no_retune` và tuning grid
rỗng. Dataset, query, relevant label và ba metric @5 cũ không đổi. Chỉ bổ sung
Category Precision@20 được yêu cầu.

Manual test 24 query:

| Hệ | Precision@5 | Hit@5 | nDCG@5 | Category P@20 |
|---|---:|---:|---:|---:|
| text-only | 0,8917 | 0,9583 | 0,9071 | — |
| image-only | 0,9083 | 0,9167 | 0,9096 | 0,8750 |
| multimodal | 0,9000 | 0,9167 | 0,9051 | 0,8729 |

So sánh Category P@20 trước/sau:

- image-only: `0,7271 → 0,8750` (`+0,1479`);
- multimodal: `0,8604 → 0,8729` (`+0,0125`).

Kết quả và từng ranking được ghi vào `data/eval/results/latest.json` và
`data/eval/results/latest_details.csv`.

### UI và Catalog

- Áp dụng token màu/spacing/radius Phase 8 cho app và bốn trang, theo hướng card
  thương mại điện tử; tên sản phẩm giới hạn hai dòng; giá, rating, review count,
  mode badge, lý do và expander score được tách rõ.
- Repository `alexpate/awesome-design-systems` chỉ được đọc như danh mục tham
  khảo design-system; không clone và không sao chép UI library.
- Catalog duyệt đủ 800 sản phẩm bằng pagination 24/40 sản phẩm mỗi trang, có tìm
  tên, category, khoảng giá, rating, sắp xếp theo giá/rating/review count và Reset.
- Search UI phân biệt thông báo: sai định dạng, quá 5 MB, file ảnh hỏng, model
  không sẵn sàng, cache mismatch và inference error; chi tiết kỹ thuật nằm trong
  expander.

Streamlit AppTest cuối:

- `app.py` và 4 file `pages/*.py`: `5/5` file có `exceptions=0`;
- Search top-K mặc định 20, options 10/20/40;
- Catalog mặc định 24 card, 34 trang; trang cuối có 8 card;
- đổi 40 sản phẩm/trang tạo đúng 20 trang;
- lọc “Tai nghe” trả 15 card; Reset trở lại 34 trang.

### Regression và fallback/offline

Các lệnh đã chạy thật:

```text
.venv\Scripts\python.exe scripts\evaluate.py
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp=data\verification\pytest-tmp
.venv\Scripts\python.exe scripts\smoke_test.py
run_project.bat --check-only
Streamlit AppTest cho app.py và 4 file pages/*.py
```

Kết quả:

- Pytest: `33 passed, 5 subtests passed`.
- Smoke: `ok=true`, 800 sản phẩm, 14.090 review, 8 category,
  `parser_mode=ollama`, LLM `ready=true`.
- Smoke latency: text `4.762,40 ms`, image `34,66 ms`, multimodal
  `4.617,22 ms`. Warning text vượt mục tiêu 3 giây được giữ nguyên; không báo
  latency PASS cho mục này.
- `run_project.bat --check-only`: `[OK] Python 3.11, dependencies, and required artifacts are ready.`
- Text-only riêng trả 20/20 kết quả Điện tử & Phụ kiện dưới 1 triệu;
  `text_score=final_score`, `image_score=N/A`. Artifact:
  `data/verification/phase8_text_only.json`.
- Với `HOPGU_OFFLINE=1`, E5 và CLIP nạp từ cache local. Endpoint Ollama không
  khả dụng được inject để kiểm tra fallback: text-only, image-only và multimodal
  đều trả 20 kết quả; parser fallback, price filter, ba score và alpha `0.8`
  đều được assert. Sau phép thử, Ollama cấu hình thật vẫn `READY`. Artifact:
  `data/verification/phase8_fallback_offline.json`.

Giới hạn còn giữ nguyên:

- `NOT VERIFIED`: máy vật lý thứ hai.
- `NOT VERIFIED`: `run_project.sh`, như lịch sử Phase 6/7.

## Phase 9 — Data Coverage & Multimodal Quality

Ngày kiểm chứng: 12/10/2026 (Asia/Saigon).

Phase 9 giữ nguyên Streamlit, `intfloat/multilingual-e5-small`,
`openai/clip-vit-base-patch32`, `qwen2.5:3b-instruct`, alpha `0.8` và toàn bộ
label trong `data/eval/queries.jsonl`. Không tune trên test set. Lịch sử Phase
1–8, bao gồm các trạng thái `BLOCKED` cũ và ghi nhận blocker đã được gỡ, vẫn
được giữ nguyên phía trên.

### Coverage trước và sau

Trước thay đổi, catalog có 800 sản phẩm/14.090 review. Kiểm tra exact keyword
cho thấy không có túi xách, túi đeo chéo hoặc balo đúng nghĩa; kết quả “balo”
duy nhất là false positive “áo ba lỗ”, còn “giày” có một phụ kiện bánh xe patin
và một đôi giày đá bóng. Artifact trước thay đổi:
`data/verification/phase9_coverage_before.json`.

API search/detail/review của Tiki tại `https://api.tiki.vn/v2` còn dữ liệu phù
hợp. Pipeline `scripts/fetch_phase9_coverage.py` lấy thêm 60 sản phẩm thật:

- 20 túi xách/túi đeo vai nữ;
- 20 balo laptop;
- 10 giày nam và 10 giày nữ;
- 129 review có nội dung, giá VND dương và URL ảnh Tiki; không có ID trùng 800
  sản phẩm cũ.

Raw supplement được lưu tại:

- `data/raw/tiki_phase9_coverage_products.csv`;
- `data/raw/tiki_phase9_coverage_details.csv`;
- `data/raw/tiki_phase9_coverage_reviews.csv`;
- `data/raw/phase9_coverage_manifest.json`.

Pipeline gặp BytePlus WAF khi dùng host web `tiki.vn/api/v2`; lỗi được ghi nhận
và không bị coi là PASS. Blocker nguồn được gỡ bằng endpoint JSON chính thức
`api.tiki.vn/v2`, không giải challenge và không đổi nguồn dữ liệu.

`prepare_dataset.py` dùng selection manifest để giữ nguyên thứ tự, ID và broad
category của toàn bộ 800 sản phẩm Phase 8, sau đó append đúng 60 ID mới. Sau
rebuild, validator xác nhận 860 sản phẩm/14.219 review, 860 ảnh local, không có
ảnh lỗi, product/review schema hợp lệ:

| Broad category | Phase 8 | Phase 9 | Delta |
|---|---:|---:|---:|
| Bách hóa | 77 | 77 | 0 |
| Làm đẹp & Sức khỏe | 112 | 112 | 0 |
| Nhà cửa & Đời sống | 72 | 72 | 0 |
| Sách | 112 | 112 | 0 |
| Thời trang nam | 112 | 122 | +10 |
| Thời trang nữ | 91 | 121 | +30 |
| Điện gia dụng | 112 | 112 | 0 |
| Điện tử & Phụ kiện | 112 | 132 | +20 |

Exact keyword/category sau rebuild:

| Coverage thực tế | Số sản phẩm | Broad category chính |
|---|---:|---|
| túi xách | 20 | Thời trang nữ |
| túi đeo chéo | 1 | Thời trang nữ |
| balo thật (đã loại “áo ba lỗ”) | 20 | Điện tử & Phụ kiện |
| giày/sneaker thật (đã loại phụ kiện/tủ/kệ) | 26 | Thời trang nam/nữ |
| tai nghe/headphone thật | 14 | Điện tử & Phụ kiện |
| kem chống nắng | 13 | Làm đẹp & Sức khỏe |
| mỹ phẩm theo keyword/subcategory | 57 | Làm đẹp & Sức khỏe |

Chi tiết toàn bộ broad category, subcategory, keyword, ví dụ sản phẩm và kết
quả validator nằm trong `data/verification/phase9_coverage_after.json`.

Text embeddings đã rebuild thành `(860, 384)` và image embeddings thành
`(860, 512)`; cả hai manifest có đúng cùng thứ tự 860 product ID. SHA-256:

- products parquet: `bf0597c94a1c04951b79cb8c5e5a1918c50d41a036efda8e8c0536ce7963372b`;
- reviews parquet: `7b80fb467f429de437fe5a597f309d9c3f80929b32e626ec0197ce4118bfb650`;
- text embeddings: `46494739aa7baee8e7f5da9654b252b49c77bb17b2a974b5164b178788c2229c`;
- image embeddings: `835f9a4324c107cb67d472b48ad1c2a606b267e144c40b2fc55f5af5bb9874f6`.

### Root cause regression Phase 7 → Phase 8

Trước khi sửa ranking, ranking Phase 7 và Phase 8 được tái tạo cho đủ 24 query;
top-5 tái tạo khớp archive 100%. Artifact per-query:
`data/verification/phase9_multimodal_regression.json`.

Nguyên nhân không được quy về một câu “do category filter”:

- `test_beauty_03` (“bao cao su…”): parser không nhận category; CLIP zero-shot
  đoán Bách hóa/Nhà cửa, hard gate loại toàn bộ 112 relevant Beauty; nDCG@5
  giảm từ `1,0` xuống `0,0`;
- `test_appliance_03` (“máy pha cà phê espresso…”): alias “cà phê” đưa query
  sang Bách hóa trước “máy pha”; hard gate loại toàn bộ 112 relevant Điện gia
  dụng; nDCG@5 giảm từ `1,0` xuống `0,0`;
- `test_home_03`: không relevant item nào bị loại; nDCG@5 giảm
  `0,86879 → 0,85393` do min-max normalization/fusion trong candidate pool;
- 21/24 query không gây regression theo phân loại artifact.

Sau fix, hai hard-gate failure đều đạt nDCG@5 `1,0` và không loại relevant
item. `test_home_03` vẫn ở `0,85393`, thấp hơn Phase 7 khoảng `0,01486`; giới
hạn này được giữ nguyên, không chỉnh số.

### Category policy và taxonomy alias

`ParsedQuery` bổ sung `category_candidates`, `category_confidence` và
`category_alias`:

- category do UI chọn hoặc alias text confidence cao: hard constraint;
- category mơ hồ confidence vừa: không hard-filter; dùng global candidate pool
  và category bonus cố định `0,03` (chọn trước lần chạy test Phase 9, không tune
  alpha/test);
- không có category text: multimodal/image-only tiếp tục dùng CLIP zero-shot
  guardrail Phase 8;
- explicit category từ text được ưu tiên hơn ảnh, nhưng alias được kiểm thử để
  không loại relevant item do taxonomy mismatch.

Alias có test: `túi xách`, `túi đeo chéo` → Thời trang nữ; `balo` → Điện tử &
Phụ kiện; `giày/sneaker` không rõ giới tính → soft pool Thời trang nam+nữ;
`giày/sneaker nam|nữ` → category tương ứng; `tai nghe/headphone` → Điện tử &
Phụ kiện; `kem chống nắng`, `mỹ phẩm`, `bao cao su` → Làm đẹp & Sức khỏe;
`máy pha cà phê/espresso` → Điện gia dụng.

Ollama thật đã chạy 11 query alias. Structured raw JSON và kết quả parser sau
validation được lưu tại `data/verification/phase9_parser_outputs.json`. Qwen
đôi lúc trả raw JSON lặp schema hoặc category sai; lớp parser deterministic đã
chặn các field không grounded và trả đúng mapping. Đây là limitation của raw
LLM, không bị che bởi badge READY.

### In-catalog và out-of-catalog image

20 ảnh Tiki held-out được tải riêng, bốn ảnh cho mỗi nhóm túi, giày, tai nghe,
mỹ phẩm và nhà cửa. Các ID này không trùng 860 ID catalog và không được dùng để
train, build embeddings hoặc tune. Manifest:
`data/eval/out_of_catalog_manifest.json`.

| Protocol | Số ảnh | Category P@20 | Top-5 category consistency |
|---|---:|---:|---:|
| IN-CATALOG SELF IMAGE | 24 | 0,9042 | không dùng metric riêng; nDCG@5 0,9513 |
| OUT-OF-CATALOG IMAGE | 20 | 0,9000 | 0,9000 |

Held-out theo nhóm: túi `1,0`, giày `1,0`, tai nghe `1,0`, nhà cửa `1,0`, mỹ
phẩm `0,5` cho cả Category P@20 và top-5 consistency. Toàn bộ prediction,
candidate category và qualitative top-5 nằm trong
`data/verification/phase9_out_of_catalog.json`.

### Evaluation Phase 7 vs Phase 8 vs Phase 9

Kết quả Phase 8 được archive trước khi chạy lại tại:

- `data/eval/results/archive/phase8_latest_20261011T180749Z.json`;
- `data/eval/results/archive/phase8_latest_details_20261011T180749Z.csv`.

Manual test vẫn là 24 query, labels không đổi, alpha vẫn `0,8`:

| Phase | Hệ | P@5 | Hit@5 | nDCG@5 | Category P@20 | Mean latency ms |
|---|---|---:|---:|---:|---:|---:|
| 7 | text-only | 0,8917 | 1,0000 | 0,9145 | — | 13,14 |
| 7 | image-only | 0,8333 | 1,0000 | 0,8692 | — | 33,20 |
| 7 | multimodal | 0,9250 | 1,0000 | 0,9433 | — | 42,76 |
| 8 | text-only | 0,8917 | 0,9583 | 0,9071 | — | 301,73 |
| 8 | image-only | 0,9083 | 0,9167 | 0,9096 | 0,8750 | 77,34 |
| 8 | multimodal | 0,9000 | 0,9167 | 0,9051 | 0,8729 | 54,23 |
| 9 | text-only | 0,9333 | 1,0000 | 0,9487 | — | 283,39 |
| 9 | image-only | 0,9500 | 0,9583 | 0,9513 | 0,9042 | 58,04 |
| 9 | multimodal | 0,9833 | 1,0000 | 0,9884 | 0,9563 | 52,31 |

Multimodal Phase 9 cao hơn cả text-only và image-only trên P@5 và nDCG@5, và
cao hơn Phase 7/8. Category P@20 multimodal cũng cao hơn baseline Phase 8.

Một bug metric thật được sửa: evaluator Phase 8 `image_only` vô tình dùng
parsed category từ text query, trong khi production image-only chỉ dùng ảnh.
Phase 9 image-only nay dùng đúng CLIP zero-shot như production; vì vậy số
image-only Phase 8/9 không hoàn toàn cùng protocol. Multimodal vẫn so sánh trực
tiếp. `data/eval/results/latest.json` ghi rõ protocol và bảng phase comparison.

### Latency profile và cache explanation

Profile một request multimodal đã warm model, dùng Ollama thật và sinh reason
cho top-5:

| Stage | ms | Tỷ trọng |
|---|---:|---:|
| parser/LLM | 1.276,90 | 27,83% |
| text embedding | 10,14 | 0,22% |
| image embedding | 24,10 | 0,53% |
| retrieval/ranking | 3,63 | 0,08% |
| explanation generation top-5 | 3.273,95 | 71,35% |

Reason chiếm phần lớn thời gian. `OllamaService` đã thêm LRU memory cache theo
đúng model + query + evidence; không đổi ranking/evidence. Cache-hit parser đo
`0,37 ms`, explanation top-5 `0,07 ms`. Cold call vẫn phải trả chi phí Ollama.
Artifact: `data/verification/phase9_latency_profile.json`.

### Regression thực chạy

Các lệnh/check đã chạy sau rebuild:

- `pytest`: lần đầu bị lỗi collection do hai thư mục cache tạm Windows khóa;
  không báo PASS. Sau khi thêm `pytest.ini` với `testpaths=tests`, chạy lại đạt
  `37 passed, 16 subtests passed`;
- `scripts/smoke_test.py`: `ok=true`, 860 sản phẩm, 14.219 review, 8 category,
  text `1.378,25 ms`, image `28,03 ms`, multimodal `1.311,78 ms`, Ollama READY;
- `run_project.bat --check-only`: exit 0 và in `[OK]`;
- Streamlit AppTest: `app.py` + 4 trang, `5/5` có `exceptions=0`. Python phát
  warning cleanup thư mục tạm bị Windows khóa sau khi artifact đã ghi; không có
  exception trong app và process exit 0;
- text-only, image-only và multimodal chạy model thật; multimodal có đủ
  `text_score`, `image_score`, `final_score`, alpha `0,8` và toàn bộ kết quả dưới
  price filter. Artifact: `data/verification/phase9_integration.json`;
- explanation top-5 khớp product ID và không có numeric claim ngoài evidence;
- FALLBACK được inject bằng endpoint Ollama không khả dụng: parser fallback,
  text search trả kết quả và giữ price filter, System Status hiện cảnh báo
  FALLBACK. Artifact: `data/verification/phase9_fallback.json`;
- ngay sau đó endpoint cấu hình thật `127.0.0.1:11434` được kiểm tra lại và
  `qwen2.5:3b-instruct` ở trạng thái READY. Artifact:
  `data/verification/phase9_ollama_ready_after_fallback.json`;
- các run model/evaluation dùng `HOPGU_OFFLINE=1`, xác nhận E5 và CLIP nạp từ
  cache local; không tải model lúc inference.

### Limitation còn lại

- held-out mỹ phẩm chỉ đạt Category P@20/top-5 consistency `0,5`; không retune
  CLIP hoặc prompt bằng tập held-out;
- `test_home_03` vẫn thấp hơn Phase 7 khoảng `0,01486`, dù aggregate Phase 9
  tốt hơn rõ rệt;
- raw Qwen JSON chưa ổn định với một số alias, nên validated parser vẫn cần
  rule guardrail;
- cache explanation chỉ giảm latency cho query/evidence lặp lại; cold Ollama
  generation vẫn là bottleneck;
- `NOT VERIFIED`: máy vật lý thứ hai và `run_project.sh`, giữ nguyên limitation
  lịch sử.

