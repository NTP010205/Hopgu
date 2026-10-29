from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def precision_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int = 5) -> float:
    if k <= 0:
        raise ValueError("k phải lớn hơn 0")
    top = ranked_ids[:k]
    return sum(item in relevant_ids for item in top) / k


def hit_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int = 5) -> float:
    return float(any(item in relevant_ids for item in ranked_ids[:k]))


def ndcg_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int = 5) -> float:
    if not relevant_ids:
        return 0.0
    dcg = sum(
        (1.0 / math.log2(rank + 2)) if product_id in relevant_ids else 0.0
        for rank, product_id in enumerate(ranked_ids[:k])
    )
    ideal_hits = min(k, len(relevant_ids))
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / ideal if ideal else 0.0


def validate_result_payload(payload: dict[str, Any]) -> None:
    if payload.get("generated_by") != "scripts/evaluate.py":
        raise ValueError("Kết quả không có provenance từ scripts/evaluate.py")
    if not payload.get("dataset", {}).get("product_ids_sha256"):
        raise ValueError("Kết quả thiếu dataset checksum")
    if payload.get("manual_test", {}).get("query_count", 0) < 1:
        raise ValueError("Kết quả không có truy vấn test thật")
    systems = payload.get("manual_test", {}).get("systems", {})
    required = {"text_only", "image_only", "multimodal"}
    if set(systems) != required:
        raise ValueError("Kết quả phải có đủ text_only, image_only, multimodal")
    for name, metrics in systems.items():
        for metric in ("precision_at_5", "hit_at_5", "ndcg_at_5", "latency_ms_mean"):
            value = metrics.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Metric {name}.{metric} không hợp lệ")
        category_metric = metrics.get("category_precision_at_20")
        if category_metric is not None and (
            not isinstance(category_metric, (int, float))
            or not math.isfinite(category_metric)
            or not 0.0 <= category_metric <= 1.0
        ):
            raise ValueError(f"Metric {name}.category_precision_at_20 không hợp lệ")


def load_evaluation_results(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_result_payload(payload)
    return payload

