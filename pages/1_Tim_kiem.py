from __future__ import annotations

import json

import streamlit as st

from src.data.loader import load_products
from src.recommender.service import CacheMismatchError, RecommendationService
from src.ui.components import (
    render_page_intro,
    render_product_card,
    render_section_heading,
    render_state_panel,
)
from src.ui.styles import APP_CSS
from src.vision.image_encoder import ImageValidationError, validate_uploaded_image


st.set_page_config(page_title="Tìm kiếm · Hợp Gu", page_icon="🔎", layout="wide")
st.markdown(APP_CSS, unsafe_allow_html=True)
render_page_intro(
    eyebrow="Tìm kiếm thông minh",
    title="Bạn đang tìm món đồ như thế nào?",
    description=(
        "Hãy kể bằng tiếng Việt, tải lên một ảnh tham chiếu hoặc dùng cả hai. "
        "Kết quả luôn đến từ catalog hiện tại và giữ nguyên thứ tự xếp hạng của hệ thống."
    ),
)


@st.cache_data(show_spinner=False)
def get_products():
    return load_products()


@st.cache_resource(show_spinner="Đang nạp mô hình tìm kiếm…")
def get_service():
    return RecommendationService()


def show_upload_error(exc: ImageValidationError) -> None:
    detail = str(exc)
    if "vượt giới hạn" in detail:
        message = "Ảnh lớn hơn 5 MB. Hãy chọn ảnh nhỏ hơn rồi thử lại."
    elif "chấp nhận" in detail or "Định dạng" in detail:
        message = "Định dạng chưa được hỗ trợ. Hãy dùng JPG, PNG hoặc WebP."
    else:
        message = "File ảnh bị lỗi hoặc không đọc được. Hãy chọn một ảnh khác."
    st.error(message)
    with st.expander("Chi tiết kỹ thuật"):
        st.code(f"ImageValidationError: {detail}")


def show_search_error(exc: Exception) -> None:
    if isinstance(exc, CacheMismatchError):
        message = "Chỉ mục ảnh không khớp catalog. Hãy tạo lại embedding cache."
    elif isinstance(exc, FileNotFoundError):
        message = "Chưa có model hoặc embedding cần thiết cho chế độ tìm kiếm này."
    elif isinstance(exc, RuntimeError):
        message = "Mô hình CLIP/E5 hiện không khả dụng. Hãy kiểm tra System Status."
    else:
        message = "Không thể xử lý yêu cầu lúc này. Hãy thử lại hoặc đổi ảnh."
    st.error(message)
    with st.expander("Chi tiết kỹ thuật"):
        st.code(f"{type(exc).__name__}: {exc}")


def show_category_decision(metadata: dict | None) -> None:
    if not metadata:
        return
    inference = metadata.get("category_inference", {})
    categories = inference.get("candidate_categories", [])
    source = inference.get("source")
    if source == "text_query":
        st.info(f"Ưu tiên danh mục từ nội dung: **{categories[0]}**.")
    elif source == "ui_filter":
        st.info(f"Đang giới hạn trong danh mục bạn chọn: **{categories[0]}**.")
    elif source == "clip_zero_shot" and inference.get("uncertain"):
        if categories:
            st.warning(
                "CLIP chưa chắc chắn về danh mục; đang mở rộng trong: "
                + ", ".join(f"**{item}**" for item in categories)
                + "."
            )
        else:
            st.warning("CLIP chưa chắc chắn về danh mục; đang tìm trên toàn catalog.")
    elif source == "clip_zero_shot":
        st.success(f"CLIP nhận diện nhóm ảnh: **{inference['primary_category']}**.")


try:
    products = get_products()
except (FileNotFoundError, ValueError) as exc:
    st.error(str(exc))
    st.stop()

categories = ["Tất cả", *sorted(products["category"].dropna().unique().tolist())]
with st.container(border=True):
    input_col, image_col = st.columns([3, 2], gap="large")
    with input_col:
        query = st.text_area(
            "Mô tả món đồ bạn cần",
            placeholder="Ví dụ: tai nghe bluetooth chống ồn dưới 1 triệu",
            height=132,
            help="Nhấn nút Gợi ý cho tôi sau khi nhập mô tả hoặc thêm ảnh.",
        )
        st.caption("Gợi ý: mô tả loại sản phẩm, nhu cầu, mức giá hoặc đặc điểm bạn ưu tiên.")
    with image_col:
        uploaded = st.file_uploader(
            "Thêm ảnh tham chiếu",
            type=["jpg", "jpeg", "png", "webp"],
            help="JPG, PNG hoặc WebP · tối đa 5 MB",
            max_upload_size=5,
        )

query_image = None
if uploaded is not None:
    try:
        query_image = validate_uploaded_image(uploaded.getvalue(), filename=uploaded.name)
    except ImageValidationError as exc:
        show_upload_error(exc)
    else:
        preview_col, note_col = st.columns([1, 2])
        with preview_col:
            st.image(query_image, width="stretch")
        with note_col:
            st.success("Ảnh tham chiếu đã sẵn sàng")
            st.caption(uploaded.name)

min_value = int(products["price_vnd"].min())
max_value = int(products["price_vnd"].max())
with st.container(border=True):
    filter_col, price_col, count_col = st.columns([1.2, 2.2, 1])
    with filter_col:
        category = st.selectbox("Danh mục", categories)
    with price_col:
        price_min, price_max = st.slider(
            "Khoảng giá (VND)",
            min_value=min_value,
            max_value=max_value,
            value=(min_value, max_value),
            step=max(1_000, (max_value - min_value) // 100),
        )
    with count_col:
        top_k = st.selectbox("Số kết quả", [10, 20, 40], index=1)
    st.caption("Bộ lọc được áp dụng trực tiếp lên kết quả thật; mặc định hiển thị 20 sản phẩm.")

search_submitted = st.button("Gợi ý cho tôi", type="primary", width="stretch")
if search_submitted:
    if not query.strip() and query_image is None:
        st.warning("Hãy nhập mô tả hoặc tải một ảnh tham chiếu.")
    else:
        search_metadata = None
        try:
            if query.strip() and query_image is not None:
                with st.spinner("Đang kết hợp nội dung và hình ảnh…"):
                    results, search_metadata = get_service().search_multimodal(
                        query,
                        query_image,
                        category=category,
                        price_min=price_min,
                        price_max=price_max,
                        top_k=top_k,
                    )
                result_badge = "MULTIMODAL"
            elif query_image is not None:
                with st.spinner("Đang tìm sản phẩm tương đồng…"):
                    results = get_service().search_image(
                        query_image,
                        category=category,
                        price_min=price_min,
                        price_max=price_max,
                        top_k=top_k,
                    )
                    search_metadata = results.attrs.get("search_metadata")
                result_badge = "IMAGE"
            else:
                with st.spinner("Đang hiểu mô tả và xếp hạng…"):
                    results = get_service().search_text(
                        query,
                        category=category,
                        price_min=price_min,
                        price_max=price_max,
                        top_k=top_k,
                    )
                result_badge = "TEXT"
        except Exception as exc:
            show_search_error(exc)
        else:
            if results.empty:
                render_state_panel(
                    title="Chưa tìm thấy lựa chọn phù hợp",
                    message="Hãy thử nới khoảng giá, chọn tất cả danh mục hoặc mô tả theo cách khác.",
                    icon="⌕",
                )
            else:
                show_category_decision(search_metadata)
                parser_payload = None
                if search_metadata is not None:
                    parser_payload = search_metadata.get("parser")
                elif result_badge == "TEXT" and "parsed_query" in results.columns:
                    parser_payload = json.loads(results.iloc[0]["parsed_query"])
                fallback_badge = None
                if parser_payload:
                    if parser_payload["parser_mode"] == "fallback":
                        st.warning("LLM không sẵn sàng; parser dự phòng vẫn đang hoạt động.")
                        fallback_badge = "FALLBACK"
                    with st.expander("Phân tích truy vấn"):
                        st.json(parser_payload)

                render_section_heading(
                    kicker="Kết quả xếp hạng",
                    title=f"{len(results)} gợi ý phù hợp",
                    description="Mở từng card để xem bằng chứng và chi tiết điểm của kết quả.",
                )
                columns = st.columns(4)
                for index, (_, product) in enumerate(results.iterrows()):
                    with columns[index % 4]:
                        render_product_card(
                            product,
                            badge=result_badge,
                            secondary_badge=fallback_badge,
                        )
else:
    render_state_panel(
        title="Sẵn sàng tìm món hợp gu",
        message="Nhập mô tả, thêm ảnh hoặc kết hợp cả hai rồi chọn Gợi ý cho tôi.",
        icon="✦",
    )
