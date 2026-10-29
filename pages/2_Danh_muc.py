from __future__ import annotations

import streamlit as st

from src.data.loader import load_products
from src.recommender.filters import apply_filters
from src.ui.components import (
    render_page_intro,
    render_product_card,
    render_section_heading,
    render_state_panel,
)
from src.ui.styles import APP_CSS


st.set_page_config(page_title="Danh mục · Hợp Gu", page_icon="🛍️", layout="wide")
st.markdown(APP_CSS, unsafe_allow_html=True)
render_page_intro(
    eyebrow="Catalog sản phẩm thật",
    title="Khám phá từng món trong kho",
    description=(
        "Tìm theo tên, lọc theo danh mục, giá và đánh giá. Catalog được phân trang để "
        "giữ trải nghiệm nhẹ, rõ và không tải toàn bộ sản phẩm cùng lúc."
    ),
)


@st.cache_data(show_spinner=False)
def get_products():
    return load_products()


try:
    products = get_products()
except (FileNotFoundError, ValueError) as exc:
    st.error(str(exc))
    st.stop()

min_price = int(products["price_vnd"].min())
max_price = int(products["price_vnd"].max())
categories = ["Tất cả", *sorted(products["category"].dropna().unique().tolist())]
st.session_state.setdefault("catalog_name", "")
st.session_state.setdefault("catalog_category", "Tất cả")
st.session_state.setdefault("catalog_price", (min_price, max_price))
st.session_state.setdefault("catalog_rating", 0.0)
st.session_state.setdefault("catalog_sort", "Mặc định")
st.session_state.setdefault("catalog_page_size", 24)
st.session_state.setdefault("catalog_page", 1)


def reset_catalog_filters() -> None:
    st.session_state["catalog_name"] = ""
    st.session_state["catalog_category"] = "Tất cả"
    st.session_state["catalog_price"] = (min_price, max_price)
    st.session_state["catalog_rating"] = 0.0
    st.session_state["catalog_sort"] = "Mặc định"
    st.session_state["catalog_page_size"] = 24
    st.session_state["catalog_page"] = 1


with st.container(border=True):
    top_left, top_right = st.columns([5, 1])
    with top_left:
        name_query = st.text_input(
            "Tìm theo tên sản phẩm",
            placeholder="Nhập tên, thương hiệu hoặc một phần tên sản phẩm",
            key="catalog_name",
        )
    with top_right:
        st.write("")
        st.button(
            "Đặt lại",
            width="stretch",
            key="catalog_reset",
            on_click=reset_catalog_filters,
        )

    category_col, price_col, rating_col = st.columns([1.2, 2, 1])
    with category_col:
        category = st.selectbox("Danh mục", categories, key="catalog_category")
    with price_col:
        price_min, price_max = st.slider(
            "Khoảng giá (VND)",
            min_value=min_price,
            max_value=max_price,
            step=max(1_000, (max_price - min_price) // 100),
            key="catalog_price",
        )
    with rating_col:
        min_rating = st.slider(
            "Rating tối thiểu",
            min_value=0.0,
            max_value=5.0,
            step=0.5,
            key="catalog_rating",
        )

    sort_col, size_col = st.columns([2, 1])
    with sort_col:
        sort_option = st.selectbox(
            "Sắp xếp",
            [
                "Mặc định",
                "Giá: thấp đến cao",
                "Giá: cao đến thấp",
                "Rating cao nhất",
                "Nhiều đánh giá nhất",
            ],
            key="catalog_sort",
        )
    with size_col:
        page_size = st.selectbox(
            "Sản phẩm mỗi trang",
            [24, 40],
            key="catalog_page_size",
        )

visible = apply_filters(
    products,
    category=category,
    price_min=price_min,
    price_max=price_max,
)
if name_query.strip():
    visible = visible.loc[
        visible["product_name"].str.contains(
            name_query.strip(),
            case=False,
            na=False,
            regex=False,
        )
    ]
visible = visible.loc[visible["rating"].astype(float).ge(min_rating)]

if sort_option == "Giá: thấp đến cao":
    visible = visible.sort_values("price_vnd", ascending=True, kind="stable")
elif sort_option == "Giá: cao đến thấp":
    visible = visible.sort_values("price_vnd", ascending=False, kind="stable")
elif sort_option == "Rating cao nhất":
    visible = visible.sort_values("rating", ascending=False, kind="stable")
elif sort_option == "Nhiều đánh giá nhất":
    visible = visible.sort_values("review_count", ascending=False, kind="stable")

filter_signature = (
    name_query.strip().casefold(),
    category,
    int(price_min),
    int(price_max),
    float(min_rating),
    sort_option,
    int(page_size),
)
if st.session_state.get("catalog_filter_signature") != filter_signature:
    st.session_state["catalog_filter_signature"] = filter_signature
    st.session_state["catalog_page"] = 1

page_count = max(1, (len(visible) + page_size - 1) // page_size)
if st.session_state.get("catalog_page", 1) > page_count:
    st.session_state["catalog_page"] = page_count

render_section_heading(
    kicker="Kết quả catalog",
    title=f"{len(visible):,} sản phẩm".replace(",", "."),
    description=f"Catalog gốc hiện có {len(products):,} sản phẩm thật.".replace(",", "."),
)


def previous_catalog_page() -> None:
    st.session_state["catalog_page"] = max(1, int(st.session_state["catalog_page"]) - 1)


def next_catalog_page() -> None:
    st.session_state["catalog_page"] = min(
        page_count,
        int(st.session_state["catalog_page"]) + 1,
    )


previous_col, page_col, next_col = st.columns([1, 1.2, 1])
with previous_col:
    st.write("")
    st.button(
        "← Trang trước",
        width="stretch",
        disabled=st.session_state["catalog_page"] <= 1,
        on_click=previous_catalog_page,
    )
with page_col:
    page = st.number_input(
        "Trang",
        min_value=1,
        max_value=page_count,
        step=1,
        key="catalog_page",
    )
with next_col:
    st.write("")
    st.button(
        "Trang sau →",
        width="stretch",
        disabled=st.session_state["catalog_page"] >= page_count,
        on_click=next_catalog_page,
    )

start = (int(page) - 1) * page_size
page_products = visible.iloc[start : start + page_size]
st.caption(f"Đang xem trang {int(page)}/{page_count} · {len(page_products)} sản phẩm trên trang")

if page_products.empty:
    render_state_panel(
        title="Không có sản phẩm phù hợp",
        message="Hãy thử đổi từ khóa, chọn tất cả danh mục hoặc nới lỏng khoảng giá và rating.",
        icon="⌕",
    )
else:
    columns = st.columns(4)
    for index, (_, product) in enumerate(page_products.iterrows()):
        with columns[index % 4]:
            render_product_card(product, badge="CATALOG")
