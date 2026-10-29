from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import PROJECT_ROOT


def format_vnd(value: int | float) -> str:
    return f"{int(value):,} ₫".replace(",", ".")


def resolve_image_path(raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def render_page_intro(*, eyebrow: str, title: str, description: str) -> None:
    st.markdown(
        (
            '<section class="hopgu-hero">'
            '<div class="hopgu-hero__content">'
            f'<div class="hopgu-eyebrow">{escape(eyebrow)}</div>'
            f'<h1>{escape(title)}</h1>'
            f'<p>{escape(description)}</p>'
            "</div></section>"
        ),
        unsafe_allow_html=True,
    )


def render_section_heading(*, kicker: str, title: str, description: str = "") -> None:
    detail = f"<p>{escape(description)}</p>" if description else ""
    st.markdown(
        (
            '<div class="hopgu-section-heading">'
            f'<div class="hopgu-kicker">{escape(kicker)}</div>'
            f'<h2>{escape(title)}</h2>{detail}'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_stat_grid(items: list[tuple[str, str, str]]) -> None:
    cards = "".join(
        (
            '<div class="hopgu-stat-card">'
            f'<div class="hopgu-stat-label">{escape(label)}</div>'
            f'<div class="hopgu-stat-value">{escape(value)}</div>'
            f'<div class="hopgu-stat-detail">{escape(detail)}</div>'
            "</div>"
        )
        for label, value, detail in items
    )
    st.markdown(f'<div class="hopgu-stat-grid">{cards}</div>', unsafe_allow_html=True)


def render_state_panel(*, title: str, message: str, icon: str = "✦") -> None:
    st.markdown(
        (
            '<div class="hopgu-state">'
            f'<div class="hopgu-state__icon">{escape(icon)}</div>'
            f'<h3>{escape(title)}</h3>'
            f'<p>{escape(message)}</p>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_status_card(*, label: str, status: str, detail: str) -> None:
    normalized = status.casefold()
    if normalized not in {"ready", "fallback", "error", "pending"}:
        normalized = "pending"
    st.markdown(
        (
            f'<div class="hopgu-status-card hopgu-status-card--{normalized}">'
            '<div class="hopgu-status-top">'
            f'<div class="hopgu-status-label">{escape(label)}</div>'
            f'<span class="hopgu-status-pill hopgu-status-pill--{normalized}">{escape(status.upper())}</span>'
            "</div>"
            f'<div class="hopgu-status-detail">{escape(detail)}</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _score_row(label: str, value: object) -> str:
    rendered = f"{float(value):.4f}" if pd.notna(value) else "—"
    return (
        '<div class="hopgu-score-row">'
        f"<span>{escape(label)}</span><strong>{escape(rendered)}</strong>"
        "</div>"
    )


def render_product_card(
    product: pd.Series,
    *,
    badge: str = "CATALOG",
    secondary_badge: str | None = None,
) -> None:
    with st.container(border=True):
        image_path = resolve_image_path(str(product["image_path"]))
        if image_path.is_file():
            st.image(str(image_path), width="stretch")
        else:
            render_state_panel(
                title="Chưa có ảnh",
                message="Ảnh của sản phẩm này hiện không đọc được.",
                icon="▧",
            )

        badges = []
        for value in (badge, secondary_badge):
            if not value:
                continue
            modifier = ""
            if value.upper() == "FALLBACK":
                modifier = " hopgu-badge--fallback"
            elif value.upper() == "CATALOG":
                modifier = " hopgu-badge--catalog"
            badges.append(f'<span class="hopgu-badge{modifier}">{escape(value.upper())}</span>')

        safe_name = escape(str(product["product_name"]))
        safe_category = escape(str(product["category"]))
        raw_review_count = product.get("review_count", 0)
        review_count = int(raw_review_count) if pd.notna(raw_review_count) else 0
        review_count_text = f"{review_count:,}".replace(",", ".")
        st.markdown(
            (
                f'<div class="hopgu-badge-row">{"".join(badges)}</div>'
                f'<div class="hopgu-category">{safe_category}</div>'
                f'<div class="hopgu-product-title">{safe_name}</div>'
                f'<div class="hopgu-price">{format_vnd(product["price_vnd"])}</div>'
                f'<div class="hopgu-meta">⭐ {float(product["rating"]):.1f}/5&nbsp;&nbsp;·&nbsp;&nbsp;'
                f'{review_count_text} đánh giá</div>'
            ),
            unsafe_allow_html=True,
        )

        reason = product.get("reason")
        if reason and not pd.isna(reason):
            with st.expander("Vì sao phù hợp?"):
                st.markdown(
                    f'<div class="hopgu-reason">{escape(str(reason))}</div>',
                    unsafe_allow_html=True,
                )

        if "final_score" in product.index:
            with st.expander("Chi tiết điểm"):
                st.markdown(
                    (
                        _score_row("Khớp nội dung", product.get("text_score"))
                        + _score_row("Khớp hình ảnh", product.get("image_score"))
                        + _score_row("Điểm tổng hợp", product.get("final_score"))
                    ),
                    unsafe_allow_html=True,
                )
