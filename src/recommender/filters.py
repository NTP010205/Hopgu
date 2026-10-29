from __future__ import annotations

import pandas as pd


def apply_filters(
    products: pd.DataFrame,
    *,
    category: str | None = None,
    categories: list[str] | tuple[str, ...] | None = None,
    price_min: int | None = None,
    price_max: int | None = None,
) -> pd.DataFrame:
    if price_min is not None and price_max is not None and price_min > price_max:
        raise ValueError("Giá tối thiểu không được lớn hơn giá tối đa")

    mask = pd.Series(True, index=products.index)
    if category and categories:
        raise ValueError("Chỉ dùng category hoặc categories, không dùng đồng thời")
    if category and category != "Tất cả":
        mask &= products["category"].astype(str).eq(category)
    elif categories:
        mask &= products["category"].astype(str).isin(categories)
    prices = pd.to_numeric(products["price_vnd"], errors="coerce")
    if price_min is not None:
        mask &= prices.ge(price_min)
    if price_max is not None:
        mask &= prices.le(price_max)
    return products.loc[mask].copy()

