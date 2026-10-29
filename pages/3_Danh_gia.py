from __future__ import annotations

import json

import streamlit as st

from src.config import EVAL_RESULTS_DIR
from src.evaluation.metrics import load_evaluation_results
from src.ui.components import render_page_intro, render_section_heading, render_state_panel
from src.ui.styles import APP_CSS


st.set_page_config(page_title="Đánh giá · Hợp Gu", page_icon="📊", layout="wide")
st.markdown(APP_CSS, unsafe_allow_html=True)
render_page_intro(
    eyebrow="Đánh giá thực nghiệm",
    title="Chất lượng được đo bằng dữ liệu thật",
    description=(
        "So sánh hệ văn bản, hình ảnh và đa phương thức trên kết quả được sinh bởi pipeline đánh giá. "
        "Trang này chỉ đọc artifact hiện tại, không tự tạo hoặc hard-code metric."
    ),
)
result_path = EVAL_RESULTS_DIR / "latest.json"
if not result_path.is_file():
    render_state_panel(
        title="Chưa có số liệu đánh giá",
        message="Chạy scripts/evaluate.py sau khi có embeddings; trang này không hiển thị metric mẫu.",
        icon="▥",
    )
    st.stop()

try:
    payload = load_evaluation_results(result_path)
except (OSError, json.JSONDecodeError, ValueError) as exc:
    st.error(f"File kết quả đánh giá không hợp lệ: {exc}")
    st.stop()

st.caption(
    f"Sinh lúc {payload['generated_at']} · alpha={payload['alpha_selection']['selected']:.1f} "
    "(giữ cố định, không retune ở Phase 8)"
)
systems = payload["manual_test"]["systems"]
render_section_heading(
    kicker="Bộ test tiếng Việt",
    title="So sánh ba chế độ tìm kiếm",
    description=f"{payload['manual_test']['query_count']} query test; alpha chỉ chọn trên validation.",
)

system_labels = {
    "text_only": "Text only",
    "image_only": "Image only",
    "multimodal": "Multimodal",
}
metric_columns = st.columns(len(systems), gap="medium")
for column, (name, values) in zip(metric_columns, systems.items(), strict=True):
    with column:
        with st.container(border=True):
            category_precision = values.get("category_precision_at_20")
            metric_items = [
                ("Precision@5", f"{values['precision_at_5']:.4f}"),
                ("Hit@5", f"{values['hit_at_5']:.4f}"),
                ("nDCG@5", f"{values['ndcg_at_5']:.4f}"),
                (
                    "Category P@20",
                    f"{category_precision:.4f}" if category_precision is not None else "—",
                ),
            ]
            cells = "".join(
                (
                    '<div class="hopgu-eval-metric">'
                    f'<div class="hopgu-eval-label">{label}</div>'
                    f'<div class="hopgu-eval-value">{value}</div>'
                    "</div>"
                )
                for label, value in metric_items
            )
            st.markdown(
                (
                    f'<div class="hopgu-eval-title">{system_labels.get(name, name)}</div>'
                    f'<div class="hopgu-eval-grid">{cells}</div>'
                    f'<div class="hopgu-eval-latency">Latency trung bình: '
                    f'{values["latency_ms_mean"]:.2f} ms</div>'
                ),
                unsafe_allow_html=True,
            )

st.caption("Bảng số liệu đầy đủ")
st.dataframe(
    [
        {
            "Hệ": name,
            "Precision@5": values["precision_at_5"],
            "Hit@5": values["hit_at_5"],
            "nDCG@5": values["ndcg_at_5"],
            "Category P@20": values.get("category_precision_at_20"),
            "Latency TB (ms)": values["latency_ms_mean"],
        }
        for name, values in systems.items()
    ],
    width="stretch",
    hide_index=True,
)
comparison = payload.get("category_precision_at_20_comparison")
if comparison:
    render_section_heading(
        kicker="Nhất quán danh mục",
        title="Trước và sau guardrail",
        description=f"Đo trên {comparison['query_count']} truy vấn ảnh thật; metric @5 vẫn giữ nguyên.",
    )
    st.dataframe(
        [
            {
                "Hệ": mode,
                "Trước": comparison["before"][mode],
                "Sau": comparison["after"][mode],
                "Thay đổi": comparison["delta"][mode],
            }
            for mode in ("image_only", "multimodal")
        ],
        width="stretch",
        hide_index=True,
    )
render_section_heading(
    kicker="Kiểm tra bổ sung",
    title="Proxy test",
    description=payload["proxy"]["note"],
)
st.dataframe(
    [{"Hệ": name, **values} for name, values in payload["proxy"]["systems"].items()],
    width="stretch",
    hide_index=True,
)
with st.expander("Provenance và tuning grid"):
    st.json(payload)

