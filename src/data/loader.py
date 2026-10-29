from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DIR
from src.data.validator import validate_products, validate_reviews


def load_products(
    path: Path | None = None,
    *,
    validate: bool = True,
    require_local_images: bool = True,
) -> pd.DataFrame:
    target = path or (PROCESSED_DIR / "products.parquet")
    if not target.is_file():
        raise FileNotFoundError(
            f"Chưa có dữ liệu sản phẩm: {target}. Hãy chạy scripts/prepare_dataset.py và scripts/download_images.py."
        )
    products = pd.read_parquet(target)
    products["product_id"] = products["product_id"].astype(str)
    if validate:
        validate_products(products, require_local_images=require_local_images).raise_for_errors()
    return products


def load_reviews(path: Path | None = None, *, validate: bool = True) -> pd.DataFrame:
    target = path or (PROCESSED_DIR / "reviews.parquet")
    if not target.is_file():
        raise FileNotFoundError(
            f"Chưa có dữ liệu review: {target}. Hãy chạy scripts/prepare_dataset.py và scripts/download_images.py."
        )
    reviews = pd.read_parquet(target)
    reviews["product_id"] = reviews["product_id"].astype(str)
    if validate:
        validate_reviews(reviews).raise_for_errors()
    return reviews

