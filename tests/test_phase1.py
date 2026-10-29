from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT
from src.data.validator import validate_products
from src.recommender.filters import apply_filters


def product_frame(image_path: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "product_id": "tiki-1",
                "product_name": "Kem chống nắng",
                "category": "Làm đẹp & Sức khỏe",
                "price_vnd": 250_000,
                "description": "Dành cho da dầu",
                "rating": 4.7,
                "review_count": 2,
                "reviews_text": ["Thấm nhanh, không bết dính"],
                "image_path": image_path,
                "source": "Tiki",
                "product_url": "https://tiki.vn/example",
            }
        ]
    )


class DatasetValidationTests(unittest.TestCase):
    def test_valid_schema_and_existing_image(self) -> None:
        report = validate_products(
            product_frame("PROMPT_CODEX_BUILD_HOPGU_V2.md"), project_root=PROJECT_ROOT
        )
        self.assertTrue(report.ok, report.errors)

    def test_duplicate_product_id_is_rejected(self) -> None:
        frame = pd.concat([product_frame("missing.jpg")] * 2, ignore_index=True)
        report = validate_products(frame, require_local_images=False)
        self.assertFalse(report.ok)
        self.assertTrue(any("trùng" in error for error in report.errors))

    def test_missing_image_is_rejected(self) -> None:
        report = validate_products(product_frame("definitely-missing.jpg"), project_root=PROJECT_ROOT)
        self.assertFalse(report.ok)
        self.assertTrue(any("Thiếu ảnh local" in error for error in report.errors))


class PriceFilterTests(unittest.TestCase):
    def test_price_bounds_are_strictly_respected(self) -> None:
        frame = pd.concat(
            [
                product_frame("a.jpg"),
                product_frame("b.jpg").assign(product_id="tiki-2", price_vnd=600_000),
            ],
            ignore_index=True,
        )
        result = apply_filters(frame, price_min=100_000, price_max=500_000)
        self.assertEqual(result["product_id"].tolist(), ["tiki-1"])
        self.assertTrue(result["price_vnd"].between(100_000, 500_000).all())

    def test_inverted_price_range_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            apply_filters(product_frame("a.jpg"), price_min=500_000, price_max=100_000)


if __name__ == "__main__":
    unittest.main()

