from __future__ import annotations

import inspect
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.nlp.llm_service import LLMStatus
from src.nlp.query_parser import QueryParser
from src.recommender.service import RecommendationService
from src.vision.category_classifier import CategoryPrediction, ZeroShotCategoryClassifier


class OfflineLLM:
    def status(self) -> LLMStatus:
        return LLMStatus(False, "fallback", "unit test offline")


class FakeClipEncoder:
    def encode_texts(self, prompts):
        vectors = []
        for prompt in prompts:
            if "book" in prompt.lower():
                vectors.append([0.0, 1.0, 0.0])
            elif "electronic" in prompt.lower() or "computer" in prompt.lower():
                vectors.append([0.0, 0.0, 1.0])
            else:
                vectors.append([1.0, 0.0, 0.0])
        return np.asarray(vectors, dtype=np.float32)

    def encode_image(self, image):
        return np.asarray(image, dtype=np.float32)


class FakeTextEncoder:
    def encode_query(self, query):
        return np.asarray([1.0, 0.0], dtype=np.float32)


class FixedCategoryClassifier:
    def predict_vector(self, image_vector):
        return CategoryPrediction(
            primary_category="Điện tử & Phụ kiện",
            candidate_categories=("Điện tử & Phụ kiện",),
            scores={"Điện tử & Phụ kiện": 0.3, "Làm đẹp & Sức khỏe": 0.2},
            confidence="high",
            margin=0.1,
            strategy="top_1_category",
        )


class CategoryAwareTests(unittest.TestCase):
    def test_zero_shot_high_confidence_uses_one_category(self) -> None:
        classifier = ZeroShotCategoryClassifier(
            FakeClipEncoder(),
            ["Bách hóa", "Sách", "Điện tử & Phụ kiện"],
        )
        prediction = classifier.predict_vector(np.asarray([1.0, 0.0, 0.0]))
        self.assertEqual(prediction.primary_category, "Bách hóa")
        self.assertEqual(prediction.candidate_categories, ("Bách hóa",))
        self.assertFalse(prediction.uncertain)

    def test_zero_shot_uncertain_prediction_does_not_claim_certainty(self) -> None:
        classifier = ZeroShotCategoryClassifier(
            FakeClipEncoder(),
            ["Bách hóa", "Sách", "Điện tử & Phụ kiện"],
        )
        prediction = classifier.predict_vector(np.asarray([1.0, 1.0, 0.0]))
        self.assertTrue(prediction.uncertain)
        self.assertEqual(prediction.strategy, "global_catalog")
        self.assertEqual(prediction.candidate_categories, ())

    def test_multimodal_text_category_and_price_are_hard_constraints(self) -> None:
        products = pd.DataFrame(
            [
                self._product("e1", "Tai nghe A", "Điện tử & Phụ kiện", 300_000),
                self._product("e2", "Tai nghe B", "Điện tử & Phụ kiện", 1_500_000),
                self._product("b1", "Kem chống nắng", "Làm đẹp & Sức khỏe", 200_000),
                self._product("b2", "Sữa rửa mặt", "Làm đẹp & Sức khỏe", 250_000),
            ]
        )
        service = RecommendationService(
            products=products,
            text_encoder=FakeTextEncoder(),
            image_encoder=FakeClipEncoder(),
            llm=OfflineLLM(),
            embeddings_dir=Path("data/verification"),
        )
        service.category_classifier = FixedCategoryClassifier()
        service._text_matrix = np.asarray(
            [[1.0, 0.0], [0.8, 0.2], [1.0, 0.0], [0.9, 0.1]],
            dtype=np.float32,
        )
        service._image_matrix = np.asarray(
            [[1.0, 0.0], [0.8, 0.2], [1.0, 0.0], [0.9, 0.1]],
            dtype=np.float32,
        )

        results, metadata = service.search_multimodal(
            "tai nghe dưới 1 triệu",
            np.asarray([1.0, 0.0], dtype=np.float32),
        )

        self.assertEqual(results["product_id"].tolist(), ["e1"])
        self.assertTrue(results["price_vnd"].le(1_000_000).all())
        self.assertEqual(metadata["category_inference"]["source"], "text_query")
        self.assertEqual(metadata["alpha"], 0.8)
        self.assertTrue(
            {"text_score", "image_score", "final_score"}.issubset(results.columns)
        )

    def test_image_only_uses_clip_category_candidates(self) -> None:
        products = pd.DataFrame(
            [
                self._product("e1", "Tai nghe A", "Điện tử & Phụ kiện", 300_000),
                self._product("b1", "Kem chống nắng", "Làm đẹp & Sức khỏe", 200_000),
            ]
        )
        service = RecommendationService(
            products=products,
            text_encoder=FakeTextEncoder(),
            image_encoder=FakeClipEncoder(),
            llm=OfflineLLM(),
            embeddings_dir=Path("data/verification"),
        )
        service.category_classifier = FixedCategoryClassifier()
        service._image_matrix = np.asarray(
            [[1.0, 0.0], [1.0, 0.0]],
            dtype=np.float32,
        )
        results = service.search_image(np.asarray([1.0, 0.0], dtype=np.float32))
        self.assertEqual(results["product_id"].tolist(), ["e1"])
        self.assertEqual(
            results.attrs["search_metadata"]["category_inference"]["strategy"],
            "top_1_category",
        )

    def test_service_defaults_to_twenty_results(self) -> None:
        for method_name in ("search_text", "search_image", "search_multimodal"):
            parameter = inspect.signature(
                getattr(RecommendationService, method_name)
            ).parameters["top_k"]
            self.assertEqual(parameter.default, 20)

    @staticmethod
    def _product(product_id: str, name: str, category: str, price: int) -> dict:
        return {
            "product_id": product_id,
            "product_name": name,
            "category": category,
            "price_vnd": price,
            "description": "",
            "rating": 4.5,
            "review_count": 1,
            "reviews_text": ["Ổn"],
            "image_path": "unused.jpg",
            "source": "test",
            "product_url": "",
            "image_url": "",
            "category_name": category,
            "brand_name": "",
        }


class Phase8ParserTests(unittest.TestCase):
    def test_fashion_terms_map_to_existing_broad_categories(self) -> None:
        parser = QueryParser(
            OfflineLLM(),
            allowed_categories=[
                "Thời trang nam",
                "Thời trang nữ",
                "Điện tử & Phụ kiện",
            ],
        )
        self.assertEqual(parser.parse("túi xách giống hình").category, "Thời trang nữ")
        self.assertEqual(parser.parse("giày thể thao giống hình").category, "Thời trang nam")
        self.assertEqual(parser.parse("balo laptop chống nước").category, "Điện tử & Phụ kiện")


if __name__ == "__main__":
    unittest.main()
