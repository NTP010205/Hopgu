#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hợp Gu V2 — Trích xuất danh mục sản phẩm từ Parquet sang JSON cho giao diện Lookbook Editorial.
"""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "products_catalog.json"

CATEGORY_SLUG_MAP = {
    "Thời trang nữ": "womens_fashion",
    "Thời trang nam": "mens_fashion",
    "Làm đẹp & Sức khỏe": "beauty_personal_care",
    "Điện tử & Phụ kiện": "electronics_accessories",
    "Nhà cửa & Đời sống": "home_living",
    "Điện gia dụng": "home_appliances",
    "Bách hóa": "grocery",
    "Sách": "books",
}

def export_catalog() -> Path:
    products_parquet = PROCESSED_DIR / "products.parquet"
    if not products_parquet.is_file():
        raise FileNotFoundError(f"Không tìm thấy file {products_parquet}")

    df = pd.read_parquet(products_parquet)
    catalog = []

    for _, row in df.iterrows():
        pid = str(row.get("product_id", "")).strip()
        cat_raw = str(row.get("category", "")).strip()
        cat_slug = CATEGORY_SLUG_MAP.get(cat_raw, "other")
        brand = str(row.get("brand_name", "")).strip() if pd.notna(row.get("brand_name")) else ""
        
        # Review snippet
        reviews_text = row.get("reviews_text")
        review_snippet = ""
        if isinstance(reviews_text, (list, tuple)) and len(reviews_text) > 0 and reviews_text[0]:
            review_snippet = str(reviews_text[0]).strip().replace("\n", " ")
            if len(review_snippet) > 160:
                review_snippet = review_snippet[:157] + "..."
        elif pd.notna(row.get("description")):
            desc = str(row["description"]).strip().replace("\n", " ")
            review_snippet = (desc[:157] + "...") if len(desc) > 160 else desc

        # Sizes
        if "Thời trang" in cat_raw:
            sizes = ["S", "M", "L", "XL"]
        elif "giày" in str(row.get("product_name", "")).lower() or "dép" in str(row.get("product_name", "")).lower():
            sizes = ["38", "39", "40", "41", "42"]
        else:
            sizes = ["Tiêu chuẩn"]

        material = f"Thương hiệu {brand}" if brand and brand.lower() != "nan" else "Chính hãng Tiki"
        
        # Image URIs: provide local image path, CDN URL, and local filename
        local_rel_path = f"data/images/{pid}.jpg"
        cdn_url = str(row.get("image_url", "")).strip() if pd.notna(row.get("image_url")) else ""
        
        item = {
            "product_id": pid,
            "product_name": str(row.get("product_name", "")).strip(),
            "category": cat_raw,
            "category_slug": cat_slug,
            "price_vnd": int(row.get("price_vnd", 0)),
            "rating": round(float(row.get("rating", 5.0)), 1),
            "review_count": int(row.get("review_count", 0)),
            "sizes": sizes,
            "material": material,
            "image_uri": f"/data/images/{pid}.jpg",
            "image_local": local_rel_path,
            "image_url": cdn_url or f"https://salt.tikicdn.com/cache/280x280/media/catalog/product/{pid}.jpg",
            "review_snippet": review_snippet,
            "product_url": str(row.get("product_url", "")).strip() if pd.notna(row.get("product_url")) else "",
            "description": str(row.get("description", "")).strip() if pd.notna(row.get("description")) else "",
        }
        catalog.append(item)

    OUTPUT_PATH.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Đã xuất {len(catalog)} sản phẩm sang {OUTPUT_PATH}")
    return OUTPUT_PATH

if __name__ == "__main__":
    export_catalog()
