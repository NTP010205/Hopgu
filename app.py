from __future__ import annotations

import streamlit as st

from src.config import PROCESSED_DIR
from src.data.loader import load_products
from src.ui.components import render_page_intro, render_section_heading, render_stat_grid
from src.ui.styles import APP_CSS


st.set_page_config(page_title="Hợp Gu V2", page_icon="✨", layout="wide")
st.markdown(APP_CSS, unsafe_allow_html=True)
render_page_intro(
    eyebrow="Gợi ý đa phương thức cho tiếng Việt",
    title="Tìm món đồ thật sự hợp gu",
    description=(
        "Mô tả điều bạn cần, thêm một ảnh tham chiếu hoặc kết hợp cả hai. "
        "Hợp Gu tìm trong catalog sản phẩm thật và giải thích vì sao từng lựa chọn phù hợp."
    ),
)

products_path = PROCESSED_DIR / "products.parquet"
if not products_path.is_file():
    st.warning("Chưa có dữ liệu đã xử lý. Hãy chạy pipeline chuẩn bị dữ liệu trong README.")
    st.stop()


@st.cache_data(show_spinner=False)
def cached_products():
    return load_products()


products = cached_products()
render_stat_grid(
    [
        ("Sản phẩm thật", f"{len(products):,}".replace(",", "."), "Ảnh, giá và metadata từ catalog hiện tại"),
        ("Danh mục", str(products["category"].nunique()), "Phủ nhiều nhóm nhu cầu mua sắm"),
        ("Nguồn dữ liệu", "Tiki", "Snapshot đã kiểm chứng, không sinh dữ liệu giả"),
    ]
)

render_section_heading(
    kicker="Bắt đầu",
    title="Một hành trình tìm kiếm gọn gàng",
    description="Năm khu vực, mỗi khu vực phục vụ một mục đích rõ ràng.",
)

columns = st.columns(5, gap="medium")
steps = [
    ("01", "Tìm kiếm", "Gợi ý bằng văn bản, hình ảnh hoặc cả hai.", "pages/1_Tim_kiem.py"),
    ("02", "Danh mục", "Duyệt 860 sản phẩm với bộ lọc và phân trang.", "pages/2_Danh_muc.py"),
    ("03", "Đánh giá", "Đọc metric thực được sinh từ pipeline hiện tại.", "pages/3_Danh_gia.py"),
    ("04", "Hệ thống", "Kiểm tra dữ liệu, embeddings và Ollama.", "pages/4_He_thong.py"),
    ("05", "Lookbook", "Runway Editorial thời trang cao cấp.", "pages/5_Lookbook_Atelier.py"),
]
for column, (number, title, detail, page) in zip(columns, steps, strict=True):
    with column:
        with st.container(border=True):
            st.markdown(
                (
                    f'<div class="hopgu-kicker">{number}</div>'
                    f'<div class="hopgu-step"><strong>{title}</strong><span>{detail}</span></div>'
                ),
                unsafe_allow_html=True,
            )
            st.page_link(page, label=f"Mở {title}")

st.caption(
    "Dữ liệu sản phẩm, giá và review là snapshot thật. Giao diện không tạo sản phẩm, "
    "điểm số hoặc kết quả đánh giá mẫu."
)

