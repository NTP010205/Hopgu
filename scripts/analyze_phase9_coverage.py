from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DIR
from src.data.validator import validate_products, validate_reviews


def fold(value: object) -> str:
    text = unicodedata.normalize("NFD", str(value).lower().replace("đ", "d"))
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def keyword_report(products: pd.DataFrame, pattern: str, exclude: str | None = None) -> dict:
    text = (products["product_name"].fillna("") + " " + products["category_name"].fillna(""))
    folded = text.map(fold)
    mask = folded.str.contains(pattern, regex=True)
    if exclude:
        mask &= ~folded.str.contains(exclude, regex=True)
    selected = products.loc[mask]
    return {
        "count": len(selected),
        "broad_categories": selected["category"].value_counts().sort_index().to_dict(),
        "subcategories": selected["category_name"].value_counts().head(10).to_dict(),
        "examples": selected[
            ["product_id", "product_name", "category_name", "category"]
        ].head(20).rename(columns={"category": "broad_category"}).to_dict("records"),
    }


def main() -> int:
    products_path = PROCESSED_DIR / "products.parquet"
    reviews_path = PROCESSED_DIR / "reviews.parquet"
    products = pd.read_parquet(products_path)
    reviews = pd.read_parquet(reviews_path)
    product_report = validate_products(products)
    review_report = validate_reviews(reviews, set(products["product_id"].astype(str)))
    product_report.raise_for_errors()
    review_report.raise_for_errors()

    patterns = {
        "túi xách": (r"\btui xach\b|\bhandbag\b", None),
        "túi đeo chéo": (r"\btui deo cheo\b|\bcrossbody\b", None),
        "balo": (r"\bbalo\b|\bba lo\b|\bbackpack\b", r"\bao ba lo\b|\bao 3 lo\b"),
        "giày/sneaker": (
            r"(?:^| )giay(?: |$)|\bsneaker\b",
            r"\bphu kien\b|\btu giay\b|\bke giay\b|\bbanh xe giay\b|\bve sinh giay\b|\bgiay lau\b|\bbinh xit\b",
        ),
        "tai nghe/headphone": (
            r"\btai nghe\b|\bheadphones?\b",
            r"\bdung cu ve sinh\b|\bphu kien\b",
        ),
        "kem chống nắng": (r"\bkem chong nang\b|\bsunscreen\b", None),
        "mỹ phẩm": (
            r"\bmy pham\b|\bkem chong nang\b|\bsua rua mat\b|\bson duong\b|\bdau goi\b|\bkem duong\b",
            None,
        ),
    }
    coverage_manifest_path = PROJECT_ROOT / "data" / "raw" / "phase9_coverage_manifest.json"
    coverage_manifest = json.loads(coverage_manifest_path.read_text(encoding="utf-8"))
    added = set(map(str, coverage_manifest["coverage_product_ids"]))
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "catalog_products": len(products),
        "catalog_reviews": len(reviews),
        "catalog_broad_categories": products["category"].value_counts().sort_index().to_dict(),
        "catalog_subcategories": [
            {
                "broad_category": broad,
                "subcategory": sub,
                "count": int(count),
            }
            for (broad, sub), count in products.groupby(
                ["category", "category_name"], dropna=False
            ).size().sort_values(ascending=False).items()
        ],
        "catalog_keywords": {
            label: keyword_report(products, pattern, exclude)
            for label, (pattern, exclude) in patterns.items()
        },
        "phase9_additions": {
            "count": int(products["product_id"].astype(str).isin(added).sum()),
            "reviews": int(reviews["product_id"].astype(str).isin(added).sum()),
            "groups": {
                key: len(value)
                for key, value in coverage_manifest["coverage_by_group"].items()
            },
        },
        "validation": {
            "products_ok": product_report.ok,
            "reviews_ok": review_report.ok,
            "product_warnings": product_report.warnings,
            "review_warnings": review_report.warnings,
        },
        "checksums": {
            "products_parquet_sha256": sha256(products_path),
            "reviews_parquet_sha256": sha256(reviews_path),
            "coverage_manifest_sha256": sha256(coverage_manifest_path),
        },
    }
    output = PROJECT_ROOT / "data" / "verification" / "phase9_coverage_after.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(output),
        "products": len(products),
        "reviews": len(reviews),
        "categories": payload["catalog_broad_categories"],
        "keywords": {
            key: value["count"] for key, value in payload["catalog_keywords"].items()
        },
        "validation": payload["validation"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
