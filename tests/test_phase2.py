from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.nlp.text_encoder import build_product_text
from src.nlp.review_analyzer import top_review_evidence
from src.recommender.scoring import cosine_scores


class TextScoringTests(unittest.TestCase):
    def test_cosine_scoring_orders_nearest_vector_first(self) -> None:
        matrix = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype=np.float32)
        scores = cosine_scores(matrix, np.asarray([1.0, 0.0], dtype=np.float32))
        self.assertEqual(int(np.argmax(scores)), 0)
        self.assertGreater(float(scores[2]), float(scores[1]))

    def test_zero_vector_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            cosine_scores(np.asarray([[1.0, 0.0]], dtype=np.float32), np.zeros(2, dtype=np.float32))

    def test_product_text_uses_real_fields_and_reviews(self) -> None:
        row = pd.Series(
            {
                "product_name": "Tai nghe Bluetooth",
                "category": "Điện tử & Phụ kiện",
                "description": "Có chống ồn chủ động",
                "reviews_text": ["Âm thanh rõ", "Pin dùng lâu"],
            }
        )
        text = build_product_text(row)
        self.assertIn("Tai nghe Bluetooth", text)
        self.assertIn("Âm thanh rõ", text)
        self.assertIn("Pin dùng lâu", text)

    def test_review_evidence_accepts_parquet_numpy_array(self) -> None:
        reviews = np.asarray(["Review thật đầu tiên", "Review thứ hai"], dtype=object)
        self.assertEqual(top_review_evidence(reviews), "Review thật đầu tiên")


if __name__ == "__main__":
    unittest.main()

