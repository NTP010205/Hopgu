from __future__ import annotations

import json
import unittest

import numpy as np

from src.nlp.llm_service import LLMStatus, OllamaService
from src.nlp.query_parser import QueryParser
from src.recommender.fusion import fuse_scores


class OfflineLLM:
    def status(self) -> LLMStatus:
        return LLMStatus(False, "fallback", "unit test: offline")


class MisleadingReadyLLM:
    def status(self) -> LLMStatus:
        return LLMStatus(True, "ollama", "unit test: ready")

    def parse_query(self, query: str, allowed_categories: list[str]) -> dict:
        return {
            "category": None,
            "colors": ["hồng"],
            "attributes": ["nam", "dưới 1 triệu"],
            "price_min": 1_000_000,
            "price_max": 10_000_000,
            "gender": "nữ",
            "keywords": ["giày chạy bộ", "không có trong query"],
        }


class EchoGroundedReasonOllama(OllamaService):
    def _chat_json(self, system: str, user: str):
        return json.loads(user)["required_output_example"]


class HallucinatingReasonOllama(OllamaService):
    def _chat_json(self, system: str, user: str):
        product_id = json.loads(user)["required_product_id"]
        return {product_id: "Sản phẩm tuyệt vời với thuộc tính không có trong evidence"}


class VietnameseQueryParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = QueryParser(llm=OfflineLLM())

    def test_required_vietnamese_queries(self) -> None:
        cases = [
            ("túi đeo chéo nữ màu đen dưới 500 nghìn", {"gender": "nữ", "price_max": 500_000}),
            ("giày chạy bộ nam nhẹ", {"gender": "nam"}),
            ("tai nghe bluetooth chống ồn", {"category": "Điện tử & Phụ kiện"}),
            ("kem chống nắng cho da dầu", {"category": "Làm đẹp & Sức khỏe"}),
            ("balo laptop chống nước", {"category": "Điện tử & Phụ kiện"}),
        ]
        for query, expected in cases:
            with self.subTest(query=query):
                parsed = self.parser.parse(query)
                self.assertEqual(parsed.parser_mode, "fallback")
                for field, value in expected.items():
                    self.assertEqual(getattr(parsed, field), value)
                self.assertTrue(parsed.keywords)

    def test_color_and_attributes_are_extracted(self) -> None:
        parsed = self.parser.parse("túi nữ màu đen chống nước dưới 500 nghìn")
        self.assertIn("đen", parsed.colors)
        self.assertIn("chống nước", parsed.attributes)

    def test_color_does_not_match_inside_another_word(self) -> None:
        parsed = self.parser.parse("tai nghe bluetooth chống ồn")
        self.assertNotIn("hồng", parsed.colors)

    def test_search_verb_does_not_become_purple(self) -> None:
        parsed = self.parser.parse("tìm sản phẩm giống hình nhưng dưới 1 triệu")
        self.assertNotIn("tím", parsed.colors)

    def test_cheaper_than_is_a_maximum_price(self) -> None:
        parsed = self.parser.parse(
            "tôi muốn một chiếc túi giống hình này nhưng rẻ hơn 600 nghìn"
        )
        self.assertIsNone(parsed.price_min)
        self.assertEqual(parsed.price_max, 600_000)

    def test_llm_fields_are_grounded_and_price_is_guarded_by_query(self) -> None:
        parser = QueryParser(llm=MisleadingReadyLLM())
        parsed = parser.parse("giày chạy bộ nam nhẹ dưới 1 triệu")
        self.assertEqual(parsed.parser_mode, "ollama")
        self.assertEqual(parsed.category, "Thể thao & Dã ngoại")
        self.assertEqual(parsed.gender, "nam")
        self.assertIsNone(parsed.price_min)
        self.assertEqual(parsed.price_max, 1_000_000)
        self.assertNotIn("hồng", parsed.colors)
        self.assertNotIn("dưới 1 triệu", parsed.attributes)
        self.assertNotIn("không có trong query", parsed.keywords)

    def test_llm_cannot_invent_category_for_generic_query(self) -> None:
        parser = QueryParser(llm=MisleadingReadyLLM())
        parsed = parser.parse("tìm sản phẩm giống hình nhưng dưới 1 triệu")
        self.assertIsNone(parsed.category)


class FusionTests(unittest.TestCase):
    def test_fusion_normalizes_modalities_before_weighting(self) -> None:
        text = np.asarray([100.0, 0.0], dtype=np.float32)
        image = np.asarray([0.0, 1.0], dtype=np.float32)
        result = fuse_scores(text, image, alpha=0.6)
        np.testing.assert_allclose(result, [0.6, 0.4], atol=1e-6)

    def test_invalid_alpha_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            fuse_scores(np.ones(2), np.ones(2), alpha=1.1)


class GroundedReasonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.evidence = [
            {
                "product_id": "p1",
                "name": "Tai nghe A",
                "category": "Điện tử & Phụ kiện",
                "price_vnd": 359_000,
                "rating": 4.7,
                "description": "",
                "review": "Hàng đúng mô tả. Bass nghe đã tai.",
                "text_score": 0.9,
                "recommendation_score": 0.8971,
            }
        ]

    def test_exact_evidence_reason_is_accepted(self) -> None:
        reasons = EchoGroundedReasonOllama().generate_reasons("tai nghe", self.evidence)
        self.assertEqual(set(reasons), {"p1"})
        self.assertIn("359000 VND", reasons["p1"])
        self.assertIn('review thật: "Hàng đúng mô tả. Bass"', reasons["p1"])

    def test_ungrounded_free_form_reason_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            HallucinatingReasonOllama().generate_reasons("tai nghe", self.evidence)


if __name__ == "__main__":
    unittest.main()

