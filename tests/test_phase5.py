from __future__ import annotations

import unittest

from src.evaluation.metrics import hit_at_k, ndcg_at_k, precision_at_k, validate_result_payload


class MetricTests(unittest.TestCase):
    def test_ranking_metrics_use_actual_ranked_ids(self) -> None:
        ranked = ["a", "b", "c", "d", "e"]
        relevant = {"b", "d"}
        self.assertEqual(precision_at_k(ranked, relevant, 5), 0.4)
        self.assertEqual(hit_at_k(ranked, relevant, 5), 1.0)
        self.assertGreater(ndcg_at_k(ranked, relevant, 5), 0.0)
        self.assertLess(ndcg_at_k(ranked, relevant, 5), 1.0)

    def test_result_without_script_provenance_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "provenance"):
            validate_result_payload({"manual_test": {"query_count": 99}})

    def test_complete_generated_payload_is_accepted(self) -> None:
        metrics = {"precision_at_5": 0.1, "hit_at_5": 0.2, "ndcg_at_5": 0.3, "latency_ms_mean": 1.0}
        payload = {
            "generated_by": "scripts/evaluate.py",
            "dataset": {"product_ids_sha256": "abc"},
            "manual_test": {
                "query_count": 1,
                "systems": {"text_only": metrics, "image_only": metrics, "multimodal": metrics},
            },
        }
        validate_result_payload(payload)


if __name__ == "__main__":
    unittest.main()

