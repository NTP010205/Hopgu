# Editorial integration — 2026-10-25

**Trạng thái: CHƯA COMPLETE.** Tích hợp code và kiểm chứng tự động đã thực hiện;
browser manual TEXT / IMAGE / MULTIMODAL, ảnh hiển thị và responsive chưa kiểm chứng.
Không dùng kết quả Node hoặc Streamlit AppTest để tự nhận browser manual PASS.

## Audit và root cause

Backend chuẩn: `D:\Thực hành\TTNTUD\project\hopgu_v2`, chỉ đọc.
Project tích hợp: thư mục chứa tài liệu này, dưới `Hop_gu_v2_da_chinh_giao_dien/hopgu_v2_submit/hopgu_v2`.

`src/recommender/service.py` cung cấp:

```python
search_text(query, *, category=None, price_min=None, price_max=None, top_k=20)
search_image(image, *, category=None, price_min=None, price_max=None, top_k=20)
search_multimodal(query, image, *, category=None, price_min=None, price_max=None,
                  top_k=20, alpha=None)  # returns (DataFrame, metadata)
```

Text/image trả DataFrame, metadata ở `attrs['search_metadata']`. Multimodal trả tuple.
Adapter không truyền alpha; `SETTINGS.fusion_alpha` trong backend hiện là 0.8.
Catalog dùng `src.data.loader.load_products()` và validator sẵn có.
Ảnh catalog resolve từ `PROJECT_ROOT / image_path`; API chỉ phục vụ basename trong
`IMAGES_DIR` sau resolve/containment, kiểm tra tồn tại và MIME.
Upload dùng nguyên hàm `validate_uploaded_image(content, filename=...)` trong
`src/vision/image_encoder.py`: 5 MB, JPG/PNG/WebP, Pillow verify và RGB decode.
Qwen/rule-based do `QueryParser`/`OllamaService` trong backend quyết định.
Review evidence qua `top_review_evidence`; reason là cột backend trả về.
Category policy, lọc giá và top_k nằm trong RecommendationService/apply_filters.
System Status gốc kiểm tra file artifact và `OllamaService.status()`;
adapter còn nạp E5/CLIP và kiểm tra manifest/matrix trước khi báo runtime READY.

Tại đầu audit, `serve_editorial.py` đã có adapter gọi service nhưng frontend chưa
nối hoàn chỉnh: `runAiSearch()` vẫn chạy local scoring, có parser category JS,
image score cố định, fusion 50/50 và popularity bonus. Ảnh upload bị gán
`bags_wallets`. Không có kết quả thì hiển thị catalog thay thế. Modal có size,
material/review mẫu; standalone và trang Lookbook có đường nhúng dữ liệu cũ.
Đó là lý do giao diện trông hoạt động dù không thực hiện search AI thật.

## Kiến trúc sau sửa

HTML/CSS/JavaScript → `serve_editorial.py` → `RecommendationService` → E5 / CLIP /
Qwen hoặc fallback nguyên bản → JSON → render cards/modal.

- GET `/api/catalog` dùng catalog thật, lọc và sắp xếp duyệt catalog trên Python.
- POST `/api/recommend` nhận multipart, giữ original File, dispatch 3 mode đúng signature.
- Score không tính lại ở adapter/JS; JSON đổi NaN thành null, giữ thứ tự ranking.
- Text/category/price và image guardrail đều giao backend xử lý.
- Product ảnh local; thiếu ảnh có UI thiếu ảnh. Không fallback ảnh sản phẩm sang Unsplash.
- HTML escape tên, category, description, reason, review; API không phục vụ arbitrary filesystem.
- Backend lỗi trả HTTP error + JSON; frontend xóa kết quả, không dùng mock products.
- `/api/status` biểu thị dịch vụ; request có thể FALLBACK dù Ollama READY nếu backend
  báo parser hoặc explanation thất bại. `llm_reason_error` được giữ và có cảnh báo UI.
- Streamlit Lookbook nhúng URL API server, không nhúng stale standalone catalog.
- CSS editorial trước/sau giống byte nội dung; giữ typography, animation và layout.
  Nội dung giới thiệu mua hàng/cửa hàng giả đã thay bằng mô tả chức năng hiện có.
- Ảnh người mẫu chỉ là artwork editorial; không xác nhận sản phẩm catalog có trong ảnh.

## Files changed/created

Trong thư mục ứng dụng:

- `serve_editorial.py`: adapter HTTP, serialization, validation, status, local images và request log.
- `hopgu_v2_editorial_frontend.html`: gọi API, upload File, cards/modal/evidence và trạng thái thật.
- `pages/5_Lookbook_Atelier.py`: iframe tới server API, không inject catalog hoặc viết standalone.
- `run_editorial.bat`: dùng Python 3.11 `.venv`, check-only, model cache offline.
- `hopgu_standalone_full.html`: hướng dẫn mở server thay cho app AI giả chạy file://.
- `h_p_gu_th_i_trang_thi_t_k_lookbook_cao_c_p.html`: entry hướng dẫn tương tự.
- `m_y_ch_t_ch_h_p_ng_g_i_to_n_b_h_p_gu.py`: delegate vào server chuẩn, không implementation thứ hai.
- `README.md`: kiến trúc/cách chạy/test/giới hạn đúng thực tế.
- `scripts/verify_editorial_api.py`: real-model HTTP checks và parity với service.
- `scripts/verify_editorial_frontend.mjs`: Node boundary test với HTTP thật; không phải browser.
- `scripts/verify_editorial_integrity.py`: so checksum trước/sau.
- `INTEGRATION_VALIDATION.md`, `integration_verification/*`: báo cáo và evidence thực tế.
- `integration_backup_20261025/*`: backup file trước sửa và manifest checksum.

Ở thư mục cha `hopgu_v2_submit`: cập nhật `run_editorial.bat` wrapper và
`hopgu_standalone_full.html` để không còn mở đường demo cũ.

Môi trường: bảo toàn venv nhập từ máy Huy thành `.venv_imported_huy_20261025`,
sao chép Python/dependency/model cache hiện có sang `.python`, `.venv`, `data/models`
của project tích hợp. Không tải dataset, không rebuild/tune embeddings.
Không sửa `src/*`, các test backend hiện có, evaluation hay VALIDATION lịch sử.

## Fake logic đã bỏ khỏi production

Không còn `textScore =`, `imgScore =`, `finalScore =`, popularity bonus,
JS parser tiếng Việt, hard-code `uploadedRefImage.category = bags_wallets`,
mock catalog hoặc fallback catalog khi kết quả rỗng/lỗi. Không còn size/material/review
tự điền trong product modal. `Math.random` còn lại chỉ phục vụ animation particles.
Backup giữ bản cũ để đối chiếu và không được server phục vụ.

## Test results đã chạy

| Kiểm tra | Kết quả và evidence |
|---|---|
| Backend pytest | PASS: 37 passed, 16 subtests passed. `integration_verification/pytest.log` |
| Smoke nguyên bản | PASS: 860 sản phẩm, 14.219 review, 8 category; image/multimodal self-query top1 `271312499`. `smoke.log` |
| `run_project.bat --check-only` | PASS, Python 3.11/dependency/artifact đầy đủ |
| `run_editorial.bat --check-only`, wrapper thư mục ngoài | PASS. `check_only.log` |
| Streamlit AppTest | PASS 6 trang, không exception. `apptest.json` |
| HTTP TEXT | PASS E5 thật, query `tai nghe bluetooth`, parser Qwen thật, 3 kết quả và score đúng backend |
| HTTP IMAGE | PASS upload JPEG bytes thật, vision validator + CLIP + search_image, top1 `271312499` |
| HTTP MULTIMODAL | PASS retrieval/score parity; alpha 0.8. Explanation fallback được ghi riêng bên dưới |
| Catalog | PASS 860 items, local image JPEG hợp lệ, category/price filter và sort |
| Category/giá multimodal | PASS category Điện tử & Phụ kiện và 100.000–1.000.000 VND, top_k=5 |
| Empty/invalid upload | PASS empty thật không catalog thay thế; byte ảnh sai, đuôi exe, >5 MB, top_k=0, giá ngược và category sai đều bị từ chối |
| Path traversal/content type | PASS request traversal, file không tồn tại/arbitrary path và JSON thay multipart bị từ chối |
| Ollama READY | PASS status thật và parser_mode=ollama, spy ghi 15 lần gọi `_chat_json` thực tế trong verifier cuối |
| Ollama unavailable | PASS endpoint không khả dụng `http://127.0.0.1:1` trong process riêng; ConnectTimeout, fallback rule-based, 0 lần `_chat_json`; E5/CLIP vẫn thực thi |
| Sau fallback | PASS server bình thường `/api/status` trở lại/giữ READY. `final_status.json` |
| JS→HTTP→render | PASS Node harness: cả 3 mode FormData, file bytes, điểm backend, modal/evidence escaping, loading/empty/error và warning fallback. `frontend_api.json` |
| Browser manual TEXT/IMAGE/MULTIMODAL | **BLOCKED / NOT VERIFIED**; không có browser/native connection |
| Browser ảnh/responsive/animation/modal | **NOT VERIFIED**; CSS được so sánh nhưng chưa kiểm tra render thực tế |

HTTP full responses và structured parser output: `api_ready.json`, `api_fallback.json`.
Method spies chỉ đếm rồi gọi hàm gốc, không cung cấp dữ liệu/model/response giả.
Verifier so toàn bộ record HTTP với `_result_records()` của kết quả gọi trực tiếp service:
IDs, rank, text_score, image_score, final_score, giá, rating, image URL, reason và review.
`server_final.log` ghi POST `/api/recommend` cho TEXT, IMAGE, MULTIMODAL và tên method thật.
`frontend_api.json` ghi trường multipart, size/tên file và response thực tế.
Không dùng các ảnh test làm training/tuning.

## Qwen explanation: limitation thực tế

Lần chạy cuối với Ollama READY:

| Mode | Parser | Request status | Top1 | Thời gian HTTP (top_k=3) |
|---|---|---|---|---:|
| TEXT | ollama | READY | 272030678 | 3184.53 ms |
| IMAGE | không dùng parser text | READY | 271312499 | 162.48 ms |
| MULTIMODAL | ollama | FALLBACK | 271312499 | 1485.63 ms |

TEXT trả reason có giá/rating/review evidence đúng dataset, ví dụ sản phẩm 272030678:
`Điểm gợi ý 0.9025; giá 359000 VND; rating 4.7/5; review thật: "Hàng đúng mô tả. Bass".`
MULTIMODAL ghi `metadata.llm_reason_error = "ValueError"`. Backend giữ reason từ review
thật; adapter không thay guardrail/LLM hoặc ép thành công. UI hiện FALLBACK kèm warning.
Đây không phải thất bại retrieval; **không báo PASS cho Qwen explanation multimodal**.
Backend text search không xuất telemetry chi tiết mọi lỗi reason; adapter giữ nguyên
giới hạn này, không suy diễn provenance LLM cho từng reason chỉ từ status READY.

Smoke warm latency: text 1508.25 ms, image 32.18 ms, multimodal 1322.47 ms.
Đây là một lần đo thực tế, không phải benchmark có kiểm soát hay metric evaluation mới.
`HOPGU_OFFLINE=1` được bật cho các inference test; không download/rebuild/tune dữ liệu.
Không kiểm chứng môi trường chặn toàn bộ egress; log thư viện có cảnh báo HF Hub.

## Lịch sử kiểm chứng và blocker

Lần đầu verifier chạy với kỳ vọng FALLBACK đã FAIL vì runtime thực tế READY;
giữ nguyên tại `initial_status_expectation_mismatch.json/.log`. Kết quả quan sát trực tiếp
được dùng thay cho suy đoán từ kiểm kê process/port trong sandbox.
Các report trước bổ sung warning giữ ở `api_ready_before_warning.json` và
`api_fallback_before_warning.json`.

Browser tools báo: `Computer Use native pipe is unavailable ... The system cannot find
the file specified. (os error 2)`. Inventory apps/browsers rỗng; thử in-app browser
báo `Browser is not available: iab`. Đã thử lại native inventory, lỗi vẫn lặp.
Node harness không render layout và không thay thế manual test.
Do đó **chưa đạt Definition of Done mục 13**, không báo COMPLETE.

Khi kết nối browser khôi phục, mở `http://127.0.0.1:8000` và thực hiện:

1. TEXT `tai nghe bluetooth`, kiểm tra POST `/api/recommend`, score/card/ảnh/modal.
2. Xóa text, upload `data/images/271312499.jpg`, kiểm tra IMAGE và file multipart.
3. Giữ ảnh, thêm text, kiểm tra MULTIMODAL, alpha 0.8 và warning explanation nếu có.
4. Duyệt catalog/filter, kiểm tra local images, modal, empty/error và responsive.
5. Kiểm tra FALLBACK trên instance dùng endpoint Ollama unavailable, rồi READY trên instance bình thường.

Server phiên bản cuối đã chạy ẩn ở localhost:8000; PID ghi tại `server_pid.txt`.
Launcher `run_editorial.bat` dùng để khởi động lại sau khi server dừng.
UI cần Internet cho CSS/font/ảnh người mẫu gốc; product data, embeddings và ảnh catalog local.

## Backend integrity

`integrity.json`: **PASS** — kiểm tra 1106 file snapshot backend gốc và 1066 file
backend/artifact được bảo vệ trong bản tích hợp:

- `source_changed = []`
- `target_protected_changed = []`
- `target_source_mismatch = []`
- `new_protected = []`
- `editorial_css_unchanged = true`

Phạm vi gồm `src/`, `data/processed/`, `data/embeddings/`, `data/images/`, toàn bộ
`data/eval/` (labels/results), cùng snapshot file gốc/scripts/tests/pages.
Không sửa dataset, model logic, E5, CLIP, Qwen/rules, category guardrail, alpha hoặc eval.
SHA-256 `data/eval/results/latest.json` giữ nguyên:
`b82b769a9f126a574c274bdc3b9808bbc93ac5859ba8f47470eed0376844a739`.
Snapshot không dùng timestamp, không tính runtime environment/bytecode/cache là source code.

---

# Final browser acceptance — 2026-10-29

Phần này bổ sung, không xóa lịch sử BLOCKED ngày 2026-10-25 ở trên. Blocker native
Computer Use đã được xử lý bằng Chrome thật đã mở trước đó và Chrome DevTools Protocol
trên chính browser engine. Các ca TEXT/IMAGE/MULTIMODAL, modal, lưu sản phẩm, navigation,
error states và responsive đã được chạy lại. Server cuối đang chạy tại
`http://127.0.0.1:8000`; `/api/status` trả `READY`, catalog 860, alpha 0.8,
Ollama `qwen2.5:3b-instruct` READY.

## A. UI/interaction fixes

- Mobile header đổi sang hai hàng ở <=639 px; nav cuộn ngang trong chính thanh nav,
  không làm document tràn ngang.
- Floating lookbook control trên mobile thu gọn còn nút 48 px, không che CTA/card.
- Header mobile không sticky; modal có giới hạn theo `100dvh`.
- Lookbook product dock không còn sticky đè nội dung; floating lookbook chỉ hiện ở runway.
- Nhãn mua hàng giả đã đổi thành `Đã Lưu`; drawer ghi rõ lưu cục bộ trong phiên,
  không phải đơn hàng hay backend persistence.
- Marketing claim đã đổi thành mô tả đúng chức năng E5 + CLIP và evidence.
- Invalid image không còn lộ `BytesIO`, địa chỉ bộ nhớ hay stack detail.
- HTTP 400 do input sai giữ nhãn Backend READY; chỉ lỗi network/5xx mới chuyển ERROR.
- Khi backend down, catalog/search stale bị xóa. Khi server trở lại,
  frontend phục hồi `READY · 860 sản phẩm thật` và tải lại catalog thật.

Chỉ file frontend tích hợp được sửa cho các điểm này. Không đổi service AI, dataset,
embedding, parser, guardrail, ranking, alpha hay evaluation.

## B. Browser tests thực tế

Artifact tổng: `integration_verification/browser_final/browser_final_acceptance.json`.

### TEXT

- Query: `tai nghe bluetooth chống ồn dưới 1 triệu`.
- UI: `TEXT · READY · Parser: ollama`.
- 9 kết quả, toàn bộ <=1.000.000 VND.
- Top 1: product `272030678`, Hoco EQ2, 359.000 VND, rating 4.7,
  `text_score = final_score = 0.8905476927757263`.
- Ảnh lấy từ `/data/images/272030678.jpg`, không dùng ảnh/demo fallback.
- Kết quả: **PASS**.

### IMAGE

- Browser lấy ảnh catalog thật `271312499.jpg` thành `File`, preview hiện,
  kích thước multipart 13.458 byte.
- UI: `IMAGE · READY`.
- Top 1 self-image: Sony WF-1000XM5 product `271312499`.
- `image_score = final_score = 1.0000001192092896`, `text_score = null`.
- API log ghi `method=search_image`, POST 200 và local image GET 200.
- Kết quả: **PASS**.

### MULTIMODAL

- Cùng ảnh `271312499.jpg` và query tiếng Việt ở trên.
- UI: `MULTIMODAL · READY · Parser: ollama · Alpha: 0.8`.
- Top 1 product `272030678`, giá 359.000 VND.
- `text_score = 0.8905476927757263`.
- `image_score = 0.6420391201972961`.
- `final_score = 0.9181126356124878`.
- Toàn bộ 9 kết quả <=1.000.000 VND; frontend chỉ render score JSON, không tính fusion.
- Reason dùng giá/rating/review thật:
  `Điểm gợi ý 0.9181; giá 359000 VND; rating 4.7/5; review thật: "Hàng đúng mô tả. Bass".`
- Ảnh: `integration_verification/browser_final/browser-final-multimodal.png`.
- Kết quả: **PASS**.

## C. API/network evidence

`integration_verification/server_final_utf8.log` ghi trực tiếp:

- `recommend mode=TEXT method=search_text ... parser=ollama ... status=READY`;
- `recommend mode=IMAGE method=search_image ... status=READY`;
- `recommend mode=MULTIMODAL method=search_multimodal ... alpha=0.8`;
- POST `/api/recommend` 200 và GET `/data/images/*.jpg` 200.

`scripts/verify_editorial_api.py --expect READY` exit 0 và kiểm tra parity toàn bộ
record HTTP với kết quả service gọi trực tiếp. Call spies ghi:
`search_text=3`, `search_image=2`, `search_multimodal=3`,
`encode_query=5`, `encode_image=5`, `_chat_json=13`.
Spies chỉ đếm rồi gọi hàm thật.

## D. Qwen READY/FALLBACK root cause

Qwen READY path đã chạy thật:

- `/api/status`: Ollama READY, model `qwen2.5:3b-instruct`;
- parser mode `ollama`;
- browser MULTIMODAL cuối trả request status READY và reason evidence hợp lệ;
- API READY verifier gọi `_chat_json` 13 lần.

Regression FALLBACK từng quan sát trong một MULTIMODAL top_k=3 không phải timeout,
parser fallback hay JavaScript. `generate_reasons` yêu cầu JSON map đúng
`product_id -> reason` và reason phải khớp evidence. Sau retry, response không đạt
schema/evidence nên backend ném `ValueError("LLM reason response không khớp product_id evidence")`.
Adapter giữ ranking thật, reason evidence an toàn và ghi `metadata.llm_reason_error=ValueError`.
Raw Qwen response bị từ chối không được persist nên không suy diễn nội dung raw.

Fallback unavailable được chạy trong process riêng với Ollama URL ở cổng local đóng:

- status `ready=false`, mode `fallback`, `ConnectTimeout`;
- parser rule-based nhận đúng category Điện tử & Phụ kiện, giá tối đa 1.000.000;
- E5 trả 3 sản phẩm thật, toàn bộ <=1.000.000;
- Streamlit System Status có warning FALLBACK, không có exception;
- artifact: `data/verification/integration_final_fallback.json`.

Sau đó server bình thường được kiểm tra lại và vẫn READY. Kết quả:
**OLLAMA READY PASS; OLLAMA FALLBACK PASS**.

## E. Modal, save, navigation và error states

- Modal desktop mở đúng product `272030678`, ảnh local, tên, giá, rating,
  evidence, review và ba score; nút đóng hoạt động.
- Modal mobile 390×844 có rect `358×636`, top 104, bottom 740/844,
  không vượt viewport. Ảnh:
  `integration_verification/browser_final/modal-390x844-cdp.png`.
- Save chuyển trạng thái `+ Lưu sản phẩm` / `✓ Đã lưu`, badge đổi và giữ trong
  localStorage phiên browser. UI không gọi đây là đơn hàng/backend persistence.
- Navigation runway/search/catalog/metrics, size guide, saved drawer và CTA catalog
  đã mở đúng view; floating lookbook tự ẩn ngoài runway.
- Empty text + no image: không request/fake product, hiện lời nhắc nhập mô tả/chọn ảnh.
- No result: query `túi xách dưới 10000 đồng` trả 0 và message rõ, Backend READY.
- Invalid image: browser file input gửi file giả JPG; API trả 400, UI hiện
  `Ảnh không hợp lệ hoặc không đọc được. Vui lòng chọn tệp JPG, PNG hoặc WebP hợp lệ.`,
  result count 0, Backend vẫn READY. Ảnh:
  `integration_verification/browser_final/invalid-image-390x844-final.png`.
- Backend unavailable thật: dừng đúng server PID, browser xóa catalog stale,
  hiện Backend ERROR và message rõ. Sau restart, browser phục hồi 860 items và READY.
  Ảnh down/recovered nằm trong `integration_verification/browser_final/`.

## F. Responsive results

| Viewport | Kết quả | Bằng chứng |
|---|---|---|
| 1920×1080 | PASS | `scrollWidth=1920`; header/hero/highlights/floating control không chồng |
| 1440×900 | PASS | search composer, upload, filters, grid và modal hiển thị đúng |
| 1366×768 | PASS | `scrollWidth=1366`; CTA và floating control tách nhau |
| 768×1024 | PASS | header/tablet hero/highlight cards không tràn |
| 390×844 home | PASS | `innerWidth=clientWidth=scrollWidth=390` |
| 390×844 search | PASS | composer/upload/filter nằm trong viewport |
| 390×844 modal | PASS | rect 358×636, không vượt viewport |

Ảnh viewport ở `integration_verification/browser_final/*.png`.

## G. Fake logic audit

Production frontend/server đã search lại các token:
`textScore`, `imgScore`, `finalScore`, `text_score`, `image_score`,
`final_score`, `bags_wallets`, `Math.random`, `mock`, `demo`, `fake`.

- Không có `bags_wallets`, mock/fake recommendation hay score assignment trong JS.
- `text_score/image_score/final_score` chỉ được adapter copy từ DataFrame backend và
  frontend escape/render.
- `Math.random` chỉ tạo particle animation, không tham gia search/ranking.
- Chữ `demo` duy nhất ở comment nói rõ ảnh product không fallback upstream/demo.

Kết quả: **PASS — không có fake recommendation logic trong production flow**.

## H. Automated regression cuối

| Kiểm tra | Kết quả |
|---|---|
| pytest | **PASS** — 37 passed, 16 subtests passed |
| smoke_test.py | **PASS** — 860 products, 14.219 reviews, 8 categories; LLM/parser Ollama |
| run_project.bat --check-only | **PASS** — exit 0 |
| Streamlit AppTest | **PASS** — 6 trang, 0 exception |
| frontend boundary verifier | **PASS** — TEXT/IMAGE/MULTIMODAL/FormData/modal/empty/network |
| API READY verifier | **PASS** — real inference + exact backend parity |
| API FALLBACK verifier tổng | **FAIL** — search modes/filter đã chạy FALLBACK đúng nhưng harness bị WinError 10053 ở request 415 cuối; không tính PASS |
| fallback chuyên dụng | **PASS** — ConnectTimeout + safe fallback |
| integrity checker tổng | **FAIL** — lý do duy nhất: `editorial_css_unchanged=false` |

Không sửa test để làm integrity checker PASS.

## I. Backend integrity

Lần chạy cuối `scripts/verify_editorial_integrity.py` kiểm 1.106 source file và
1.066 protected file. Các trường bảo vệ backend/dữ liệu đều **PASS**:

- `source_changed=[]`;
- `target_protected_changed=[]`;
- `target_source_mismatch=[]`;
- `new_protected=[]`;
- SHA-256 `data/eval/results/latest.json` vẫn là
  `b82b769a9f126a574c274bdc3b9808bbc93ac5859ba8f47470eed0376844a739`.

Script tổng exit 1 vì baseline cũ còn bắt buộc CSS byte-identical. Lượt này yêu cầu
sửa responsive/sticky nên `editorial_css_unchanged=false` là thay đổi frontend có chủ đích.
Không thay đổi `src/`, `data/processed/`, `data/embeddings/`, `data/images/`,
`data/eval/`, E5, CLIP, Qwen, parser, ranking, alpha hoặc metric.

## J. Acceptance matrix và final status

| Mục | Trạng thái |
|---|---|
| TEXT browser | PASS |
| IMAGE browser | PASS |
| MULTIMODAL browser | PASS |
| FILTERS | PASS |
| MODAL | PASS |
| SAVE | PASS — client/session only |
| NAVIGATION | PASS |
| RESPONSIVE DESKTOP | PASS |
| RESPONSIVE TABLET | PASS |
| RESPONSIVE MOBILE | PASS |
| OLLAMA READY | PASS |
| OLLAMA FALLBACK | PASS |
| PYTEST | PASS |
| SMOKE | PASS |
| CHECK-ONLY | PASS |
| STREAMLIT APPTEST | PASS |
| PROTECTED BACKEND/DATA INTEGRITY | PASS |
| LEGACY AGGREGATE INTEGRITY SCRIPT | FAIL — expected CSS delta |

**FINAL STATUS: NOT COMPLETE.**

Toàn bộ chức năng và Definition of Done về browser/backend/data đã được kiểm chứng,
nhưng không ghi COMPLETE vì checker integrity tổng vẫn exit 1 và verifier FALLBACK
tổng không kết thúc sạch ở request 415 cuối. Integrity fail do invariant CSS
byte-identical mâu thuẫn với yêu cầu sửa responsive; luồng fallback được verifier
chuyên dụng xác nhận PASS. Checker/test không bị sửa để ép PASS.

---

# Final Acceptance Hotfix — 2026-10-29

Phần này bổ sung và supersede trạng thái `NOT COMPLETE` ngay phía trên; toàn bộ lịch sử
trước hotfix vẫn được giữ nguyên. Phạm vi hotfix chỉ gồm hai verifier đang FAIL.
Không sửa frontend editorial, API adapter, `RecommendationService`, source AI hoặc dữ liệu.

## 1. Legacy integrity checker

### Root cause trước hotfix

Checker cũ đã kiểm checksum backend/data đúng, nhưng ghép presentation vào điều kiện fail:

```python
report['ok'] = not (source_changed or target_changed or not_matching_source or new_protected) and css_unchanged
```

Do responsive CSS được sửa có chủ đích, `css_unchanged=false` làm toàn bộ checker exit 1,
dù các danh sách protected đều rỗng. Assertion cuối là:

```python
assert report['ok'], 'Protected files changed'
```

Thông báo này không đúng phạm vi vì không có protected file nào thay đổi.

### Hotfix

File sửa: `scripts/verify_editorial_integrity.py`.

- Giữ checksum nghiêm ngặt cho canonical source và các root:
  `src/`, `data/processed/`, `data/embeddings/`, `data/images/`, `data/eval/`.
- Vẫn fail khi có file protected đổi, lệch source, bị thêm mới hoặc canonical source đổi.
- CSS/frontend được ghi rõ là `presentation_only`, `changes_allowed=true`, nhưng không
  tham gia kết quả backend integrity.
- Bổ sung semantic contract bắt buộc:
  860 products, 14.219 reviews, E5/CLIP/Qwen đúng tên, alpha 0.8.
- Bổ sung adapter contract: dùng `RecommendationService`, dispatch đủ TEXT/IMAGE/MULTIMODAL
  và dùng validator ảnh thật.
- Không bỏ checksum, không sửa baseline và không sửa protected data để vượt test.

Command:

```powershell
.\.venv\Scripts\python.exe -B .\scripts\verify_editorial_integrity.py
```

Kết quả sau hotfix: **PASS, exit 0**.

- `source_files_checked=1106`;
- `protected_files_checked=1066`;
- `source_changed=[]`;
- `target_protected_changed=[]`;
- `target_source_mismatch=[]`;
- `new_protected=[]`;
- `runtime_contract_ok=true`;
- `adapter_contract_ok=true`;
- `latest_evaluation_sha256=b82b769a9f126a574c274bdc3b9808bbc93ac5859ba8f47470eed0376844a739`;
- `ok=true`.

Artifact: `integration_verification/integrity.json`.

## 2. API FALLBACK verifier / HTTP 415

### Root cause trước hotfix

Request gây lỗi là lệnh cuối của verifier:

```python
requests.post(base + '/api/recommend', json={'query': 'x'}, timeout=10)
```

Endpoint `/api/recommend` chỉ nhận `multipart/form-data`; vì vậy HTTP 415 và code
`UNSUPPORTED_MEDIA_TYPE` là response **đúng**, không phải hành vi sai của API.

Điều tra thực tế:

- 30/30 request JSON độc lập tới server hiện hành trả 415 đúng payload;
- `/api/status` ngay sau chuỗi request vẫn trả 200;
- full FALLBACK verifier nguyên trạng cũng PASS ở lần tái hiện kế tiếp;
- server không crash, recommendation backend không đóng bất thường;
- WinError 10053 trước đó là transport reset không ổn định của Windows/test harness ở
  một request cuối, không phải lỗi ranking, fallback hoặc response contract.

### Hotfix

File sửa: `scripts/verify_editorial_api.py`.

- Gửi body JSON và `Content-Type: application/json` rõ ràng bằng fresh Session.
- Dùng `Connection: close` để không tái sử dụng socket từ các upload boundary test trước.
- Bắt buộc kiểm đủ status 415, `status=error`, code `UNSUPPORTED_MEDIA_TYPE`.
- Bắt buộc gọi `/api/status` sau response/lỗi transport để phát hiện server crash.
- Chỉ retry tối đa ba lần khi có `requests.ConnectionError`; status/payload sai vẫn fail ngay.
- Ghi số attempt, payload và server health vào artifact; không nới expected result.
- Không cần sửa `serve_editorial.py` hay recommendation backend.

Command FALLBACK:

```powershell
$env:HOPGU_OFFLINE='1'
$env:HOPGU_OLLAMA_URL='http://127.0.0.1:9'
.\.venv\Scripts\python.exe -B .\scripts\verify_editorial_api.py --expect FALLBACK
```

Kết quả stability sau hotfix: **PASS 3/3 lượt, đều exit 0**.

| Run | TEXT | IMAGE | MULTIMODAL | alpha | HTTP 415 | Attempts | Server alive |
|---:|---|---|---|---:|---|---:|---|
| 1 | PASS | PASS | PASS | 0.8 | PASS | 1 | PASS |
| 2 | PASS | PASS | PASS | 0.8 | PASS | 1 | PASS |
| 3 | PASS | PASS | PASS | 0.8 | PASS | 1 | PASS |

Artifacts:

- `integration_verification/api_fallback_stability_1.json`;
- `integration_verification/api_fallback_stability_2.json`;
- `integration_verification/api_fallback_stability_3.json`;
- `integration_verification/api_fallback.json` (latest run).

## 3. Regression sau hotfix

| Command/check | Output thực tế | Kết quả |
|---|---|---|
| `python -m pytest -q` | `37 passed, 16 subtests passed in 0.75s` | PASS |
| `scripts/smoke_test.py` | 860 products, 14.219 reviews, 8 categories; parser Ollama | PASS |
| `run_project.bat --check-only` | Python 3.11, dependencies, artifacts ready | PASS |
| `scripts/verify_editorial_frontend.mjs` | TEXT/IMAGE/MULTIMODAL/FormData/modal/empty/error đều true | PASS |
| `verify_editorial_api.py --expect READY` | real inference parity; TEXT/IMAGE/MULTIMODAL; alpha 0.8; HTTP 415 + health | PASS |
| `verify_editorial_api.py --expect FALLBACK` ×3 | 3/3 exit 0; TEXT/IMAGE/MULTIMODAL; 415; server alive | PASS |
| `verify_editorial_integrity.py` | 1.106 source + 1.066 protected; contract/checksum sạch | PASS |

API READY verifier tiếp tục ghi call thật:

- `search_text=3`;
- `search_image=2`;
- `search_multimodal=3`;
- `encode_query=5`;
- `encode_image=5`;
- `_chat_json=14`.

## 4. Files changed by this hotfix

- `scripts/verify_editorial_integrity.py`;
- `scripts/verify_editorial_api.py`;
- `INTEGRATION_VALIDATION.md`;
- runtime evidence trong `integration_verification/`.

Không thay `hopgu_v2_editorial_frontend.html` trong hotfix này. Không thay
`serve_editorial.py`, `src/`, dataset, images, embeddings hoặc evaluation.

## 5. Before / after

| Mục | Trước | Sau |
|---|---|---|
| Legacy integrity | FAIL vì CSS delta | PASS; CSS được phân loại presentation-only, protected checksum vẫn strict |
| HTTP 415 verifier | Một lần WinError 10053 ở request cuối | PASS 3/3; status/code/health được kiểm protocol rõ ràng |
| Backend/data integrity | Protected scope sạch nhưng aggregate exit 1 | Protected scope sạch và checker exit 0 |
| TEXT/IMAGE/MULTIMODAL | PASS | Vẫn PASS |

## 6. Remaining limitations

- Qwen có thể bị guardrail chuyển riêng một explanation sang reason evidence fallback nếu
  output không khớp schema; đây là behavior bảo vệ đã biết, không làm đổi ranking.
- Log có thể cảnh báo Hugging Face unauthenticated; model/cache local vẫn chạy và test exit 0.
- Frontend editorial vẫn phụ thuộc CDN cho Tailwind/font/artwork, như đã ghi ở lịch sử trước.

Không còn FAIL chưa giải thích hoặc chưa kiểm chứng trong Definition of Done của hotfix.

## 7. Final status

**FINAL STATUS: COMPLETE.**

Hai verifier còn FAIL đã được sửa đúng phạm vi và PASS thực tế; regression cũ vẫn PASS;
canonical backend Hợp Gu V2 và toàn bộ dataset/embedding/image/evaluation giữ nguyên.
