# Kế hoạch triển khai Hợp Gu V2

Ngày bắt đầu: 2026-10-11 (Asia/Saigon)

## Nguyên tắc kiểm chứng

- Xây mới hoàn toàn trong thư mục này; không sao chép kiến trúc project cũ.
- Dữ liệu demo phải là sản phẩm/review thật từ nguồn Tiki công khai, có ghi nguồn và checksum.
- Mỗi phase chỉ được đánh dấu hoàn tất sau khi chạy các lệnh kiểm thử tương ứng.
- Unit test có thể dùng fixture tối đa 20 bản ghi; dữ liệu fixture không được dùng làm catalog demo.
- Không ghi metric mẫu. Trang đánh giá chỉ đọc kết quả do `scripts/evaluate.py` sinh ra.
- Trạng thái chưa kiểm chứng được ghi rõ `BLOCKED` hoặc `CHƯA CHẠY`, không ghi PASS.

## Khảo sát môi trường ban đầu

- Thư mục ban đầu chỉ có `PROMPT_CODEX_BUILD_HOPGU_V2.md`.
- Python hiện có: CPython 3.12.2; chưa phát hiện Python 3.11.
- Streamlit hiện có ở môi trường hệ thống: 1.58.0.
- Ollama: chưa phát hiện trong `PATH`.
- Ổ D còn đủ dung lượng để chứa môi trường, model, ảnh và embeddings.
- Nguồn dữ liệu ứng viên: snapshot Tiki công khai gồm bảng sản phẩm và review tiếng Việt liên kết bằng product ID. Script chuẩn bị dữ liệu sẽ lưu URL nguồn, thời điểm tải và SHA-256.

## Phase 1 — Skeleton, schema, 50 sản phẩm thật, UI tối thiểu

Trạng thái: `COMPLETE — VERIFIED`

1. Tạo cấu trúc thư mục/module theo đặc tả.
2. Định nghĩa schema sản phẩm/review và validator.
3. Viết pipeline nhập snapshot Tiki, chọn 50 sản phẩm thật có dữ liệu hợp lệ cho lần kiểm tra đầu.
4. Tạo UI Streamlit tối thiểu đọc dữ liệu đã xử lý, có catalog và bộ lọc cơ bản.
5. Viết/running test schema, duplicate ID, missing image, filter giá và smoke import.

Điều kiện qua phase: dữ liệu 50 sản phẩm được tạo từ nguồn thật, validator qua, test Phase 1 qua và Streamlit health check trả lời.

## Phase 2 — Full dataset, ảnh local, text embeddings

Trạng thái: `COMPLETE — VERIFIED`

1. Chuẩn hóa 800–1500 sản phẩm, tối thiểu 5 nhóm danh mục, ưu tiên cân bằng.
2. Ghép review thật theo product ID; lưu `products.parquet` và `reviews.parquet`.
3. Tải ảnh snapshot về local, kiểm tra Pillow và loại record ảnh lỗi.
4. Tạo text document từ tên + danh mục + mô tả + review đã giới hạn.
5. Tạo multilingual text embeddings theo lô, kèm manifest product ID/checksum/model.
6. Bổ sung text-only search và test scoring/query tiếng Việt.

Điều kiện qua phase: validator full dataset qua; embedding manifest khớp dataset; năm truy vấn tiếng Việt chạy trên encoder thật hoặc được ghi `BLOCKED` nếu model chưa tải được.

## Phase 3 — Image embeddings và image search

Trạng thái: `COMPLETE — VERIFIED`

1. Tạo image encoder CLIP/OpenCLIP và cache resource.
2. Precompute image embeddings theo lô; manifest chỉ dùng product ID/checksum tương đối.
3. Validate upload JPG/PNG/WebP, tối đa 5 MB, RGB resize trước inference.
4. Thêm image-only search và test scoring/upload validation.

Điều kiện qua phase: một ảnh query thật được encode, search trả về sản phẩm từ dataset, cache reload thành công và integration test không mock model qua.

## Phase 4 — LLM parser và multimodal fusion

Trạng thái: `COMPLETE WITH VERIFIED FALLBACK` — multimodal/fusion/parser fallback đã kiểm chứng; nhánh Ollama thật `BLOCKED` vì máy không có server/model.

1. Tích hợp Ollama bằng HTTP local, JSON schema chặt chẽ, không cho phép sinh dữ liệu sản phẩm.
2. Hoàn thiện parser rule-based khi Ollama không sẵn sàng và hiển thị trạng thái fallback.
3. Chuẩn hóa riêng text/image score rồi fusion; không cộng vector khác không gian.
4. Sinh lý do từ trường dữ liệu đã truy xuất; LLM chỉ diễn đạt trên evidence.
5. Test parser, fusion, filter và ba chế độ text/image/multimodal.

Điều kiện qua phase: multimodal search chạy với encoder thật; trạng thái LLM thật/fallback được báo đúng, không giả nhận Ollama sẵn sàng.

## Phase 5 — Evaluation và tuning alpha

Trạng thái: `COMPLETE — VERIFIED`

1. Tạo proxy set và tối thiểu 30 query tiếng Việt có relevance được lưu công khai.
2. Chia validation/test; chọn alpha trên validation, không chọn trên test.
3. Đo Precision@5, Hit@5, nDCG@5 và latency cho text/image/multimodal bằng script.
4. Ghi kết quả có timestamp, dataset checksum, model, alpha vào `latest.json`/CSV.
5. Trang Đánh giá chỉ đọc file thật; thiếu file thì hiện “Chưa có số liệu đánh giá”.

Điều kiện qua phase: file kết quả được sinh bởi lệnh thực tế và kiểm tra chống metric hard-code qua.

## Phase 6 — UI hoàn chỉnh, one-click run, QA

Trạng thái: `COMPLETE ON WINDOWS — VERIFIED`; `run_project.sh` và máy vật lý thứ hai chưa được kiểm chứng, xem `VALIDATION.md`.

1. Hoàn thiện bốn trang Streamlit, card kết quả, badge mode, score breakdown và status.
2. Hoàn thiện `run_project.bat`, `run_project.sh`, `.env.example`, README và hướng dẫn offline.
3. Chạy toàn bộ unit/integration tests, smoke service, Streamlit health check.
4. Chạy clean-environment check bằng Python 3.11 nếu có thể cài trong workspace.
5. Tạo `VALIDATION.md` từ kết quả lệnh thực tế, ghi rõ mọi blocker còn lại.

Điều kiện hoàn tất: đáp ứng Definition of Done; mọi phần chưa thể kiểm chứng được ghi `BLOCKED`, không suy diễn PASS.

