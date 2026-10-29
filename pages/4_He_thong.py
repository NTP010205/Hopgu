from __future__ import annotations

import json

import streamlit as st

from src.config import EMBEDDINGS_DIR, PROCESSED_DIR, SETTINGS
from src.nlp.llm_service import OllamaService
from src.ui.components import render_page_intro, render_section_heading, render_status_card
from src.ui.styles import APP_CSS


st.set_page_config(page_title="Hệ thống · Hợp Gu", page_icon="🧭", layout="wide")
st.markdown(APP_CSS, unsafe_allow_html=True)
render_page_intro(
    eyebrow="System status",
    title="Mọi thành phần, trong một cái nhìn",
    description=(
        "Kiểm tra dataset, review, embeddings và kết nối Ollama đang phục vụ ứng dụng. "
        "READY, FALLBACK và ERROR được phân biệt rõ, không suy đoán trạng thái."
    ),
)

checks = {
    "Dataset": ((PROCESSED_DIR / "products.parquet").is_file(), "Catalog sản phẩm đã xử lý"),
    "Review": ((PROCESSED_DIR / "reviews.parquet").is_file(), "Review gắn với catalog hiện tại"),
    "Text embeddings": (
        (EMBEDDINGS_DIR / "text_embeddings.npy").is_file(),
        "Vector ngữ nghĩa cho E5",
    ),
    "Image embeddings": (
        (EMBEDDINGS_DIR / "image_embeddings.npy").is_file(),
        "Vector hình ảnh cho CLIP",
    ),
}
render_section_heading(
    kicker="Artifacts",
    title="Dữ liệu và chỉ mục",
    description="Trạng thái được đọc trực tiếp từ các file bắt buộc của project.",
)
artifact_columns = st.columns(2, gap="medium")
for index, (label, (ready, detail)) in enumerate(checks.items()):
    with artifact_columns[index % 2]:
        render_status_card(
            label=label,
            status="READY" if ready else "ERROR",
            detail=detail if ready else f"Không tìm thấy: {detail.casefold()}.",
        )

render_section_heading(
    kicker="Models",
    title="Mô hình đang cấu hình",
    description="Tên model được đọc từ cấu hình runtime hiện tại.",
)
model_columns = st.columns(3, gap="medium")
model_rows = [
    ("Text encoder", SETTINGS.text_model),
    ("Vision encoder", SETTINGS.vision_model),
    ("LLM parser", SETTINGS.ollama_model),
]
for column, (label, model) in zip(model_columns, model_rows, strict=True):
    with column:
        render_status_card(label=label, status="CONFIGURED", detail=model)

render_section_heading(
    kicker="Runtime",
    title="Kết nối LLM / Ollama",
    description="Nếu Ollama không sẵn sàng, parser rule-based vẫn giữ ứng dụng hoạt động.",
)
llm_status = OllamaService().status()
render_status_card(
    label="LLM / Ollama",
    status="READY" if llm_status.ready else "FALLBACK",
    detail=llm_status.detail,
)
if not llm_status.ready:
    st.warning("Ứng dụng đang dùng parser rule-based và sẽ gắn badge FALLBACK; không báo LLM sẵn sàng.")

manifest_path = PROCESSED_DIR / "dataset_manifest.json"
if manifest_path.is_file():
    with st.expander("Dataset manifest"):
        st.json(json.loads(manifest_path.read_text(encoding="utf-8")))

