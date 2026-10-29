# PROMPT CODEX — BUILD MỚI HỢP GU V2

Bạn là kỹ sư phần mềm/AI chịu trách nhiệm xây dựng mới hoàn toàn project môn **Trí tuệ nhân tạo ứng dụng** với đề tài:

**Hệ gợi ý sản phẩm thương mại điện tử đa phương thức**

## 0. NGUYÊN TẮC GỐC

Không sửa chữa chắp vá project cũ. Hãy tạo project mới trong thư mục `hopgu_v2/`.
Project cũ chỉ được dùng làm tài liệu tham khảo về ý tưởng/giao diện; không copy kiến trúc React + backend API cũ nếu nó làm tăng rủi ro chạy demo.

Project mới phải bám sát yêu cầu giảng viên:

- Mục tiêu: kết hợp text (mô tả sản phẩm, review) với hình ảnh sản phẩm để gợi ý bằng LLM kết hợp vision encoder.
- Ý tưởng chính: LLM hiểu ngôn ngữ trong review và liên kết với đặc trưng hình ảnh để tăng độ chính xác.
- Ứng dụng hướng tới: Shopee, Tiki, Lazada.
- Ứng dụng phải dùng một trong Tkinter / Flask / Streamlit.
- Chương trình bắt buộc chạy được.

Chọn **Streamlit + Python 3.11** để giảm rủi ro tích hợp frontend/backend và tránh lỗi mất kết nối khi upload ảnh.

## 1. MỤC TIÊU SẢN PHẨM

Ứng dụng phải hỗ trợ:

1. Tìm sản phẩm bằng câu tiếng Việt tự nhiên.
2. Tìm sản phẩm bằng ảnh tham chiếu.
3. Tìm bằng cả text + ảnh.
4. Lọc theo danh mục và khoảng giá VND.
5. Trả Top-K sản phẩm có:
   - ảnh;
   - tên;
   - giá;
   - rating;
   - đoạn review thật;
   - điểm text;
   - điểm image;
   - điểm multimodal cuối;
   - lý do gợi ý.
6. Có trang Catalog.
7. Có trang Đánh giá với metric thật, không hard-code.
8. Có trang System Status để biết LLM/text encoder/vision encoder/cache đã sẵn sàng hay chưa.
9. Giao diện tiếng Việt.
10. Chạy trong một process Streamlit; không dùng React + API server riêng.

## 2. KIẾN TRÚC BẮT BUỘC

Cấu trúc đề xuất:

hopgu_v2/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
├── run_project.bat
├── run_project.sh
├── pages/
│   ├── 1_Tim_kiem.py
│   ├── 2_Danh_muc.py
│   ├── 3_Danh_gia.py
│   └── 4_He_thong.py
├── src/
│   ├── config.py
│   ├── data/
│   │   ├── loader.py
│   │   ├── schema.py
│   │   └── validator.py
│   ├── nlp/
│   │   ├── query_parser.py
│   │   ├── text_encoder.py
│   │   ├── review_analyzer.py
│   │   └── llm_service.py
│   ├── vision/
│   │   └── image_encoder.py
│   ├── recommender/
│   │   ├── fusion.py
│   │   ├── filters.py
│   │   └── service.py
│   ├── evaluation/
│   │   └── metrics.py
│   └── ui/
│       ├── components.py
│       └── styles.py
├── scripts/
│   ├── prepare_dataset.py
│   ├── download_images.py
│   ├── build_text_embeddings.py
│   ├── build_image_embeddings.py
│   ├── evaluate.py
│   └── smoke_test.py
├── data/
│   ├── raw/
│   ├── processed/
│   │   ├── products.parquet
│   │   └── reviews.parquet
│   ├── images/
│   ├── embeddings/
│   └── eval/
│       └── results/
└── tests/

## 3. DATASET

Không sử dụng Amazon làm dataset chính.

Ưu tiên dữ liệu thương mại điện tử Việt Nam.

### Phương án chính
Dùng dữ liệu Tiki đã thu thập công khai/phục vụ học thuật:
- sản phẩm thật;
- giá VND;
- mô tả;
- rating;
- review tiếng Việt;
- liên kết product_id.

Chuẩn hóa thành schema:

product_id: str
product_name: str
category: str
price_vnd: int
description: str
rating: float
review_count: int
reviews_text: list[str]
image_path: str
source: str
product_url: str | None

Chỉ đưa sản phẩm vào tập cuối nếu:
- có tên;
- có danh mục;
- có giá;
- có ảnh hợp lệ;
- có mô tả hoặc review;
- product_id duy nhất.

Mục tiêu bản demo:
- 800–1500 sản phẩm;
- tối thiểu 5 danh mục;
- mỗi danh mục tương đối cân bằng;
- ảnh lưu local;
- review thật lưu local;
- không phụ thuộc Internet khi demo.

Nếu dataset Tiki hiện có không chứa ảnh, viết script tạo snapshot ảnh cho tập sản phẩm đã chọn. Không crawl quy mô lớn. Không crawl lúc chạy ứng dụng.

### Phương án dự phòng
Có thể dùng dataset Shopee Đông Nam Á có trường ảnh + mô tả + giá + rating, lọc sản phẩm Việt Nam/VND nếu xác định được.
Không dùng Shopee làm nguồn review-text duy nhất nếu dataset chỉ có review count mà không có nội dung review.

## 4. XỬ LÝ TIẾNG VIỆT

Tìm kiếm tiếng Việt phải là chức năng chính, không phải fallback phụ.

Dùng multilingual sentence embedding, ví dụ:
`intfloat/multilingual-e5-base`
hoặc model multilingual tương đương phù hợp tài nguyên.

Quy tắc:
- query tiếng Việt phải encode trực tiếp;
- product text gồm tên + danh mục + mô tả + review summary;
- không dịch query sang tiếng Anh bằng từ điển thủ công làm đường chính;
- keyword fallback chỉ dùng khi model không sẵn sàng và phải hiện cảnh báo.

Tạo test tiếng Việt tối thiểu:
- "túi đeo chéo nữ màu đen dưới 500 nghìn"
- "giày chạy bộ nam nhẹ"
- "tai nghe bluetooth chống ồn"
- "kem chống nắng cho da dầu"
- "balo laptop chống nước"

## 5. LLM

LLM có vai trò:
1. phân tích query tiếng Việt thành JSON có cấu trúc;
2. nhận diện:
   - intent;
   - category;
   - colors;
   - attributes;
   - price_min;
   - price_max;
   - gender;
   - keywords;
3. tóm tắt tín hiệu review thật;
4. tạo lý do gợi ý dựa trên dữ liệu đã truy xuất.

Ưu tiên Ollama + model local hỗ trợ tiếng Việt như `qwen2.5:3b-instruct` hoặc model tương đương.

LLM KHÔNG được:
- bịa thuộc tính sản phẩm;
- bịa review;
- bịa giá;
- tự tạo sản phẩm không có trong dataset.

Nếu Ollama không chạy:
- dùng parser rule-based;
- ứng dụng vẫn chạy;
- giao diện phải hiển thị rõ chế độ fallback;
- demo chính thức nên chạy LLM.

## 6. VISION ENCODER

Dùng CLIP/OpenCLIP hoặc vision encoder phù hợp.

Bắt buộc:
- precompute embedding cho toàn bộ ảnh sản phẩm;
- query chỉ encode đúng 1 ảnh upload;
- không đọc/hash toàn bộ ảnh dataset trên mỗi request;
- cache gắn với stable product_id + checksum dataset;
- không đưa absolute path của máy vào fingerprint;
- cache phải portable giữa thư mục/máy.

Image upload:
- JPG/PNG/WebP;
- tối đa 5 MB;
- validate bằng Pillow;
- resize/convert RGB trước inference;
- lỗi ảnh phải hiển thị cụ thể, không hiện chung "mất kết nối máy chủ".

## 7. MULTIMODAL RANKING

Tính riêng:
text_score
image_score

Chuẩn hóa score rồi fusion:

final_score = alpha * text_score + (1 - alpha) * image_score

Nếu chỉ có text:
final_score = text_score

Nếu chỉ có image:
final_score = image_score

Nếu có cả hai:
dùng fusion.

Không cộng trực tiếp vector từ hai encoder khác không gian.

Alpha:
- mặc định ban đầu 0.6 text / 0.4 image;
- sau đó chọn bằng validation;
- không chọn alpha bằng test set.

Áp filter category/price trước hoặc sau scoring theo cách có test rõ ràng nhưng tuyệt đối không trả sản phẩm vi phạm khoảng giá.

## 8. REVIEW

Review phải là dữ liệu thật.

Tạo `review_text` cho embedding bằng:
- top review hữu ích;
- hoặc concatenate có giới hạn;
- hoặc summary bằng LLM được cache trước.

Không gọi LLM để tóm tắt hàng nghìn review ở mỗi lần tìm kiếm.

Precompute review summary nếu cần.

## 9. ĐÁNH GIÁ

Không hard-code metric.

Tạo `scripts/evaluate.py`.

Đánh giá ít nhất 3 hệ:
1. text-only;
2. image-only;
3. multimodal.

Metric:
- Precision@5;
- Hit@5;
- nDCG@5;
- latency trung bình.

Tạo tập đánh giá:
A. proxy test:
- query từ title/description/image của sản phẩm;
- sản phẩm cùng category là relevant;
- ghi rõ đây là proxy.

B. manual Vietnamese query set:
- tối thiểu 30 query tiếng Việt;
- file `data/eval/queries.jsonl`;
- mỗi query có relevant_product_ids hoặc relevant_categories được gán thủ công.

Kết quả phải ghi:
`data/eval/results/latest.json`
và/hoặc CSV.

Trang Đánh giá đọc file này.
Nếu chưa chạy evaluation:
hiện "Chưa có số liệu đánh giá".

## 10. GIAO DIỆN STREAMLIT

Thiết kế sạch, hiện đại, dễ demo.

Trang tìm kiếm:
- logo Hợp Gu;
- text area;
- image uploader;
- category;
- price range;
- top_k;
- nút Tìm sản phẩm;
- preview ảnh;
- spinner tiến trình;
- kết quả dạng cards/grid.

Mỗi card:
- ảnh;
- tên;
- giá VND;
- rating;
- score;
- lý do;
- expandable score breakdown.

Có badge:
- TEXT
- IMAGE
- MULTIMODAL
- FALLBACK

Không dùng animation nặng gây crash.

## 11. PERFORMANCE

Không thực hiện công việc nặng trong mỗi lần rerun Streamlit.

Dùng:
- `st.cache_resource` cho model;
- `st.cache_data` cho dataset;
- embeddings precomputed;
- numpy matrix/vectorized cosine similarity.

Mục tiêu:
- text query: < 3 giây sau warmup;
- image/multimodal query: < 5 giây sau warmup trên máy demo;
- không có timeout HTTP vì không có frontend-backend network riêng.

## 12. ONE-CLICK RUN

`run_project.bat`:
1. kiểm tra Python 3.11;
2. tạo `.venv`;
3. install requirements nếu cần;
4. kiểm tra data processed + embeddings;
5. không tự rebuild model/cache mỗi lần chạy;
6. chạy:
   `streamlit run app.py`
7. báo lỗi rõ ràng nếu thiếu file/model.

Project phải chạy offline sau khi:
- dependencies;
- model;
- images;
- embeddings
đã được chuẩn bị.

## 13. TEST

Viết test tối thiểu cho:
- dataset schema;
- duplicate ID;
- missing image;
- Vietnamese query parsing;
- price filter;
- text scoring;
- image scoring;
- fusion;
- image upload validation;
- no fake evaluation;
- smoke test application services.

Không fake model output trong integration test cuối; unit test có thể mock dependency nặng.

## 14. PHÂN CÔNG NHÓM

Giữ cơ cấu:

TV1 — Backend / NLP / LLM
- query parser;
- LLM;
- text encoder;
- recommendation service integration.

TV2 — Backend / Vision / Ranking
- image encoder;
- cache;
- multimodal fusion;
- evaluation.

TV3 — Frontend
- Streamlit UI chính;
- cards;
- search flow;
- catalog.

TV4 — Frontend + Data
- dataset pipeline;
- image snapshot;
- schema;
- data validation;
- hỗ trợ catalog/filter UI.

TV5 — Report + Slide + QA
- báo cáo;
- slide;
- test checklist;
- ảnh minh họa;
- ghi kết quả evaluation từ file thật.

## 15. THỨ TỰ TRIỂN KHAI

Phase 1:
- tạo skeleton;
- data schema;
- load 50 product sample;
- UI tối thiểu.

Phase 2:
- chuẩn hóa full dataset;
- ảnh local;
- text embeddings.

Phase 3:
- image embeddings;
- image search.

Phase 4:
- LLM query parser;
- multimodal fusion.

Phase 5:
- evaluation;
- tuning alpha.

Phase 6:
- UI hoàn chỉnh;
- one-click run;
- tests;
- clean-machine test.

Không làm giao diện cầu kỳ trước khi text search + image search + multimodal search hoạt động.

## 16. DEFINITION OF DONE

Project chỉ được coi là xong nếu:

- chạy Streamlit thành công;
- query tiếng Việt trả kết quả hợp lý;
- upload ảnh không crash;
- image-only chạy;
- text-only chạy;
- text+image chạy;
- dataset Việt Nam thay Amazon;
- giá là VND;
- review là tiếng Việt/thật;
- LLM có vai trò thực tế;
- vision encoder có vai trò thực tế;
- score multimodal có thể giải thích;
- không metric giả;
- evaluation sinh từ script;
- app không cần Internet khi demo sau khi chuẩn bị;
- README có hướng dẫn setup;
- máy sạch chạy được theo hướng dẫn.

## 17. CÁCH CODEX LÀM VIỆC

1. Không sửa project cũ.
2. Tạo `hopgu_v2/`.
3. Lập `IMPLEMENTATION_PLAN.md` trước.
4. Triển khai từng phase.
5. Sau mỗi phase chạy test.
6. Không tuyên bố PASS nếu chưa chạy.
7. Không tạo dữ liệu giả để che lỗi.
8. Nếu dataset chưa sẵn sàng, dùng tối đa 20 record fixture chỉ cho unit test; không dùng fixture làm dữ liệu demo cuối.
9. Khi có blocker, ghi `BLOCKED` và lý do.
10. Cuối cùng tạo `VALIDATION.md` ghi lệnh thực tế đã chạy và kết quả thật.

BẮT ĐẦU BẰNG:
- khảo sát môi trường;
- tạo skeleton mới;
- viết IMPLEMENTATION_PLAN.md;
- triển khai Phase 1;
- tiếp tục các phase nếu test qua.
