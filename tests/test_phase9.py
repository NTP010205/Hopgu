from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.nlp.llm_service import LLMStatus, OllamaService
from src.nlp.query_parser import QueryParser
from src.recommender.service import CATEGORY_SOFT_BONUS, RecommendationService


class OfflineLLM:
    def status(self) -> LLMStatus:
        return LLMStatus(False, "fallback", "phase9 unit test offline")


class FakeTextEncoder:
    def encode_query(self, query):
        return np.asarray([1.0, 0.0], dtype=np.float32)


class FakeImageEncoder:
    def encode_texts(self, prompts):
        return np.eye(len(prompts), dtype=np.float32)

    def encode_image(self, image):
        return np.asarray(image, dtype=np.float32)


class CountingOllama(OllamaService):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def _chat_json(self, system: str, user: str):
        import json

        self.calls += 1
        request = json.loads(user)
        if "required_output_example" in request:
            return request["required_output_example"]
        return {
            "intent": "search_product",
            "category": None,
            "colors": [],
            "attributes": [],
            "price_min": None,
            "price_max": None,
            "gender": None,
            "keywords": [],
        }


CATEGORIES = [
    "Bách hóa",
    "Làm đẹp & Sức khỏe",
    "Nhà cửa & Đời sống",
    "Thể thao & Dã ngoại",
    "Thời trang nam",
    "Thời trang nữ",
    "Điện gia dụng",
    "Điện tử & Phụ kiện",
]


class Phase9TaxonomyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = QueryParser(OfflineLLM(), allowed_categories=CATEGORIES)

    def test_explicit_aliases_map_only_to_catalog_taxonomy(self) -> None:
        cases = {
            "túi xách công sở": "Thời trang nữ",
            "túi đeo chéo da": "Thời trang nữ",
            "balo chống nước": "Điện tử & Phụ kiện",
            "tai nghe không dây": "Điện tử & Phụ kiện",
            "headphone chống ồn": "Điện tử & Phụ kiện",
            "kem chống nắng da dầu": "Làm đẹp & Sức khỏe",
            "mỹ phẩm dưỡng da": "Làm đẹp & Sức khỏe",
            "bao cao su siêu mỏng": "Làm đẹp & Sức khỏe",
            "máy pha cà phê espresso": "Điện gia dụng",
            "giày sneaker nam": "Thời trang nam",
            "giày sneaker nữ": "Thời trang nữ",
        }
        for query, expected in cases.items():
            with self.subTest(query=query):
                parsed = self.parser.parse(query)
                self.assertEqual(parsed.category, expected)
                self.assertEqual(parsed.category_confidence, "high")
                self.assertEqual(parsed.category_candidates, [expected])

    def test_ambiguous_shoes_use_two_soft_categories(self) -> None:
        parsed = self.parser.parse("giày sneaker màu trắng")
        self.assertEqual(parsed.category_confidence, "medium")
        self.assertEqual(
            parsed.category_candidates,
            ["Thời trang nam", "Thời trang nữ"],
        )

    def test_medium_confidence_does_not_hard_filter_global_candidate(self) -> None:
        products = pd.DataFrame(
            [
                self._product("other", "Sản phẩm khác", "Làm đẹp & Sức khỏe"),
                self._product("men", "Giày nam", "Thời trang nam"),
                self._product("women", "Giày nữ", "Thời trang nữ"),
            ]
        )
        service = RecommendationService(
            products=products,
            text_encoder=FakeTextEncoder(),
            image_encoder=FakeImageEncoder(),
            llm=OfflineLLM(),
            embeddings_dir=Path("data/verification"),
        )
        service._text_matrix = np.asarray(
            [[1.0, 0.0], [0.90, 0.43589], [0.89, 0.45596]], dtype=np.float32
        )
        results = service.search_text("giày sneaker", top_k=3)
        self.assertEqual(set(results["product_id"]), {"other", "men", "women"})
        metadata = results.attrs["search_metadata"]
        self.assertEqual(
            metadata["category_policy"]["strategy"],
            "global_with_category_bonus",
        )
        self.assertEqual(metadata["candidate_count"], 3)
        self.assertAlmostEqual(
            float(results.loc[results["product_id"].eq("men"), "final_score"].iloc[0]),
            float(results.loc[results["product_id"].eq("men"), "text_score"].iloc[0])
            + CATEGORY_SOFT_BONUS,
            places=6,
        )

    def test_llm_parser_and_evidence_reason_cache_do_not_repeat_calls(self) -> None:
        llm = CountingOllama()
        first = llm.parse_query("tai nghe", CATEGORIES)
        second = llm.parse_query("tai nghe", CATEGORIES)
        self.assertEqual(first, second)
        self.assertEqual(llm.calls, 1)
        evidence = [{
            "product_id": "p1",
            "name": "Tai nghe A",
            "category": "Điện tử & Phụ kiện",
            "price_vnd": 300_000,
            "rating": 4.5,
            "review": "âm thanh tốt pin bền dùng ổn định",
            "recommendation_score": 0.9,
        }]
        reasons_first = llm.generate_reasons("tai nghe", evidence)
        reasons_second = llm.generate_reasons("tai nghe", evidence)
        self.assertEqual(reasons_first, reasons_second)
        self.assertEqual(llm.calls, 2)

    @staticmethod
    def _product(product_id: str, name: str, category: str) -> dict:
        return {
            "product_id": product_id,
            "product_name": name,
            "category": category,
            "price_vnd": 300_000,
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


if __name__ == "__main__":
    unittest.main()
