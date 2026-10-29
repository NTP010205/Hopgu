from __future__ import annotations

import json
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

from src.config import PROJECT_ROOT, PROCESSED_DIR
from src.ui.components import render_page_intro, render_section_heading
from src.ui.styles import APP_CSS

st.set_page_config(page_title="Lookbook Atelier · Hợp Gu", page_icon="✨", layout="wide")
st.markdown(APP_CSS, unsafe_allow_html=True)

render_page_intro(
    eyebrow="Editorial & Runway Lookbook",
    title="Thời trang thiết kế Hợp Gu Atelier",
    description=(
        "Không gian Lookbook cao cấp lấy cảm hứng từ sàn diễn thời trang quốc tế. "
        "Tích hợp trọn bộ 860 sản phẩm thật, bộ sưu tập Runway tương tác, gợi ý phối đồ và giỏ hàng thông minh."
    ),
)

frontend_file = PROJECT_ROOT / "hopgu_v2_editorial_frontend.html"
alt_frontend_file = PROJECT_ROOT / "h_p_gu_th_i_trang_thi_t_k_lookbook_cao_c_p.html"
standalone_file = PROJECT_ROOT / "hopgu_standalone_full.html"
catalog_file = PROCESSED_DIR / "products_catalog.json"

target_frontend = frontend_file if frontend_file.is_file() else alt_frontend_file

if not target_frontend.is_file():
    st.error(f"Không tìm thấy file giao diện Lookbook tại {PROJECT_ROOT}.")
    st.stop()

# Đảm bảo có sẵn bản standalone hoàn chỉnh
@st.cache_data(show_spinner=False)
def get_bundled_html() -> str:
    if standalone_file.is_file():
        return standalone_file.read_text(encoding="utf-8")
    
    html_text = target_frontend.read_text(encoding="utf-8")
    if catalog_file.is_file():
        catalog_data = json.loads(catalog_file.read_text(encoding="utf-8"))
        injected = (
            "<script>\n"
            f"window.__HOPGU_INJECTED_CATALOG__ = {json.dumps(catalog_data, ensure_ascii=False)};\n"
            "</script>\n</head>"
        )
        html_text = html_text.replace("</head>", injected, 1)
        standalone_file.write_text(html_text, encoding="utf-8")
    return html_text

html_content = get_bundled_html()

# Toolbar điều khiển
render_section_heading(
    kicker="Tùy chọn hiển thị",
    title="Trải nghiệm Lookbook trực tiếp hoặc toàn màn hình",
    description="Bạn có thể tương tác ngay trong khung nhìn bên dưới, hoặc mở máy chủ độc lập.",
)

col1, col2, col3 = st.columns([1.5, 1.5, 1.2], gap="small")

with col1:
    st.link_button(
        "🌐 Mở máy chủ http://127.0.0.1:8000",
        url="http://127.0.0.1:8000",
        help="Khởi chạy serve_editorial.py hoặc run_editorial.bat để truy cập cổng máy chủ đầy đủ.",
        use_container_width=True,
    )

with col2:
    if standalone_file.is_file():
        with open(standalone_file, "rb") as f:
            st.download_button(
                "💾 Tải file HTML độc lập (Offline)",
                data=f.read(),
                file_name="hopgu_standalone_full.html",
                mime="text/html",
                use_container_width=True,
            )

with col3:
    st.caption("✦ Tích hợp 860 sản phẩm thật & Runway 360°")

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

# Nhúng trực tiếp giao diện Lookbook Atelier vào Streamlit
components.html(html_content, height=920, scrolling=True)
