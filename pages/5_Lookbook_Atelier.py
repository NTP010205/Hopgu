"""Embed the HTTP application so every request retains the API's origin."""
import streamlit as st
import streamlit.components.v1 as components
from src.ui.components import render_page_intro
from src.ui.styles import APP_CSS

st.set_page_config(page_title="Lookbook Atelier · Hợp Gu", page_icon="✨", layout="wide")
st.markdown(APP_CSS, unsafe_allow_html=True)
render_page_intro(eyebrow="Editorial & Runway Lookbook", title="Hợp Gu Atelier",
                  description="Giao diện editorial kết nối RecommendationService, E5, CLIP và Qwen/Ollama.")
st.caption("Chạy run_editorial.bat trước khi mở Lookbook. Ảnh người mẫu là ảnh cảm hứng editorial; sản phẩm và gợi ý lấy từ catalog thật.")
st.link_button("Mở Lookbook toàn màn hình", "http://127.0.0.1:8000")
components.iframe("http://127.0.0.1:8000", height=920, scrolling=True)
