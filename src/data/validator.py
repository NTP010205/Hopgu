from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config import PROJECT_ROOT
from src.data.schema import PRODUCT_REQUIRED_COLUMNS, REVIEW_REQUIRED_COLUMNS


class DatasetValidationError(ValueError):
    """Dữ liệu không đạt điều kiện để đưa vào catalog."""


@dataclass(slots=True)
class ValidationReport:
    row_count: int
    categories: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if self.errors:
            raise DatasetValidationError("; ".join(self.errors))


def _missing_columns(frame: pd.DataFrame, required: Iterable[str]) -> list[str]:
    return sorted(set(required) - set(frame.columns))


def _blank_count(series: pd.Series) -> int:
    return int(series.fillna("").astype(str).str.strip().eq("").sum())


def validate_products(
    products: pd.DataFrame,
    *,
    project_root: Path = PROJECT_ROOT,
    require_local_images: bool = True,
) -> ValidationReport:
    report = ValidationReport(row_count=len(products), categories=0)
    missing = _missing_columns(products, PRODUCT_REQUIRED_COLUMNS)
    if missing:
        report.errors.append(f"Thiếu cột sản phẩm: {', '.join(missing)}")
        return report

    if products.empty:
        report.errors.append("Bảng sản phẩm rỗng")
        return report

    report.categories = int(products["category"].nunique(dropna=True))
    duplicate_count = int(products["product_id"].astype(str).duplicated().sum())
    if duplicate_count:
        report.errors.append(f"Có {duplicate_count} product_id trùng")

    for column, label in (
        ("product_id", "mã"),
        ("product_name", "tên"),
        ("category", "danh mục"),
        ("image_path", "đường dẫn ảnh"),
        ("source", "nguồn"),
    ):
        count = _blank_count(products[column])
        if count:
            report.errors.append(f"Có {count} sản phẩm thiếu {label}")

    prices = pd.to_numeric(products["price_vnd"], errors="coerce")
    invalid_price = int((prices.isna() | (prices <= 0)).sum())
    if invalid_price:
        report.errors.append(f"Có {invalid_price} sản phẩm có giá không hợp lệ")

    ratings = pd.to_numeric(products["rating"], errors="coerce")
    invalid_rating = int((ratings.isna() | ~ratings.between(0, 5)).sum())
    if invalid_rating:
        report.errors.append(f"Có {invalid_rating} sản phẩm có rating ngoài [0, 5]")

    no_text = products.apply(
        lambda row: not str(row["description"] or "").strip()
        and not _normalise_reviews(row["reviews_text"]),
        axis=1,
    )
    if int(no_text.sum()):
        report.errors.append(f"Có {int(no_text.sum())} sản phẩm thiếu cả mô tả lẫn review")

    if require_local_images:
        missing_images: list[str] = []
        for product_id, raw_path in products[["product_id", "image_path"]].itertuples(index=False):
            candidate = Path(str(raw_path))
            if not candidate.is_absolute():
                candidate = project_root / candidate
            if not candidate.is_file():
                missing_images.append(str(product_id))
        if missing_images:
            preview = ", ".join(missing_images[:5])
            report.errors.append(
                f"Thiếu ảnh local cho {len(missing_images)} sản phẩm (ví dụ: {preview})"
            )

    if report.categories < 5:
        report.warnings.append(f"Dataset hiện chỉ có {report.categories} danh mục")
    return report


def _normalise_reviews(value: object) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    return [text] if text else []


def validate_reviews(reviews: pd.DataFrame, product_ids: set[str] | None = None) -> ValidationReport:
    report = ValidationReport(row_count=len(reviews), categories=0)
    missing = _missing_columns(reviews, REVIEW_REQUIRED_COLUMNS)
    if missing:
        report.errors.append(f"Thiếu cột review: {', '.join(missing)}")
        return report
    if reviews.empty:
        report.errors.append("Bảng review rỗng")
        return report
    duplicates = int(reviews["review_id"].astype(str).duplicated().sum())
    if duplicates:
        report.errors.append(f"Có {duplicates} review_id trùng")
    empty_content = _blank_count(reviews["content"])
    if empty_content:
        report.errors.append(f"Có {empty_content} review không có nội dung")
    if product_ids is not None:
        unknown = ~reviews["product_id"].astype(str).isin(product_ids)
        if int(unknown.sum()):
            report.errors.append(f"Có {int(unknown.sum())} review không liên kết sản phẩm")
    return report

