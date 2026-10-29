from __future__ import annotations

import copy
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SETTINGS
from src.nlp.llm_service import OllamaService
from src.nlp.review_analyzer import top_review_evidence
from src.recommender.service import RecommendationService


PHASE_LABEL = os.environ.get("HOPGU_VERIFICATION_PHASE", "phase7")
OUTPUT_PATH = PROJECT_ROOT / f"data/verification/{PHASE_LABEL}_integration.json"
TEXT_QUERY = "tai nghe bluetooth chống ồn"
MULTIMODAL_QUERY = "tìm sản phẩm giống hình nhưng dưới 1 triệu"
TOP_K = 5
ALLOWED_EVIDENCE_FIELDS = {
    "product_id",
    "name",
    "category",
    "price_vnd",
    "rating",
    "description",
    "review",
    "text_score",
    "image_score",
    "recommendation_score",
}


class CapturingOllamaService(OllamaService):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.reason_calls: list[dict[str, Any]] = []
        self.raw_reason_payloads: list[Any] = []

    def _chat_json(self, system: str, user: str) -> Any:
        payload = super()._chat_json(system, user)
        if "lý do" in system:
            self.raw_reason_payloads.append(copy.deepcopy(payload))
        return payload

    def generate_reasons(
        self, query: str, evidence: list[dict[str, Any]]
    ) -> dict[str, str]:
        raw_start = len(self.raw_reason_payloads)
        record: dict[str, Any] = {
            "query": query,
            "evidence": copy.deepcopy(evidence),
        }
        try:
            reasons = super().generate_reasons(query, evidence)
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["raw_llm_json"] = copy.deepcopy(
                self.raw_reason_payloads[raw_start:]
            )
            self.reason_calls.append(record)
            raise
        record["reasons"] = copy.deepcopy(reasons)
        record["raw_llm_json"] = copy.deepcopy(
            self.raw_reason_payloads[raw_start:]
        )
        self.reason_calls.append(record)
        return reasons


def _json_value(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    columns = [
        "product_id",
        "product_name",
        "category",
        "price_vnd",
        "rating",
        "image_path",
        "text_score",
        "image_score",
        "final_score",
        "reason",
        "parser_mode",
        "parser_detail",
        "parsed_query",
    ]
    present = [column for column in columns if column in frame.columns]
    return [
        {key: _json_value(value) for key, value in record.items()}
        for record in frame[present].to_dict(orient="records")
    ]


def _numeric_tokens(value: object) -> set[str]:
    tokens = set()
    for match in re.findall(r"\d+(?:[.,]\d+)*", str(value)):
        normalized = re.sub(r"[.,]", "", match).lstrip("0") or "0"
        tokens.add(normalized)
    return tokens


def _traceability(call: dict[str, Any]) -> dict[str, Any]:
    evidence_by_id = {
        str(item["product_id"]): item for item in call["evidence"]
    }
    checks = []
    for product_id, reason in call["reasons"].items():
        item = evidence_by_id[product_id]
        allowed_numbers = _numeric_tokens(call["query"])
        allowed_numbers.add("5")  # thang rating cố định được runtime render thành /5
        for value in item.values():
            allowed_numbers.update(_numeric_tokens(value))
        allowed_numbers.update(
            _numeric_tokens(f"{float(item['recommendation_score']):.4f}")
        )
        reason_numbers = _numeric_tokens(reason)
        checks.append(
            {
                "product_id": product_id,
                "reason": reason,
                "numeric_claims": sorted(reason_numbers),
                "ungrounded_numeric_claims": sorted(reason_numbers - allowed_numbers),
                "has_real_review_evidence": bool(item.get("review")),
                "evidence_fields": sorted(item),
            }
        )
    return {
        "product_count": len(evidence_by_id),
        "reason_count": len(call["reasons"]),
        "product_ids_match": set(evidence_by_id) == set(call["reasons"]),
        "items": checks,
    }


def _assert_evidence_matches_results(
    call: dict[str, Any], results: pd.DataFrame, *, multimodal: bool
) -> None:
    expected_ids = results["product_id"].astype(str).tolist()
    actual_ids = [str(item["product_id"]) for item in call["evidence"]]
    assert actual_ids == expected_ids
    assert set(call["reasons"]) == set(expected_ids)
    assert len(call["evidence"]) == TOP_K

    rows = {str(row.product_id): row for row in results.itertuples(index=False)}
    for item in call["evidence"]:
        assert set(item).issubset(ALLOWED_EVIDENCE_FIELDS)
        row = rows[str(item["product_id"])]
        assert item["name"] == str(row.product_name)
        assert item["category"] == str(row.category)
        assert item["price_vnd"] == int(row.price_vnd)
        assert item["rating"] == float(row.rating)
        assert item["description"] == str(row.description).strip()
        assert item["review"] == top_review_evidence(row.reviews_text)
        assert item["text_score"] == float(row.text_score)
        assert item["recommendation_score"] == float(row.final_score)
        if multimodal:
            assert item["image_score"] == float(row.image_score)


def main() -> None:
    llm = CapturingOllamaService(timeout=120.0)
    status = llm.status()
    if not status.ready:
        raise RuntimeError(f"Ollama chưa READY: {status.detail}")

    service = RecommendationService(llm=llm)
    candidates = service.products[
        service.products["price_vnd"].le(1_000_000)
        & service.products["image_path"].map(
            lambda value: (PROJECT_ROOT / str(value)).is_file()
        )
    ].copy()
    candidates["has_description"] = candidates["description"].map(
        lambda value: bool(str(value).strip())
    )
    candidates = candidates.sort_values(
        ["has_description", "review_count"], ascending=False
    )
    if candidates.empty:
        raise AssertionError("Không có ảnh sản phẩm thật dưới 1 triệu")
    image_product = candidates.iloc[0]
    image_path = PROJECT_ROOT / str(image_product["image_path"])

    started = time.perf_counter()
    text_results = service.search_text(TEXT_QUERY, top_k=TOP_K)
    text_latency_ms = (time.perf_counter() - started) * 1000
    assert len(text_results) == TOP_K
    assert set(text_results["parser_mode"]) == {"ollama"}
    assert len(llm.reason_calls) == 1
    text_reason_call = copy.deepcopy(llm.reason_calls[0])
    if "error" in text_reason_call:
        print(json.dumps(text_reason_call, ensure_ascii=False, indent=2))
        raise AssertionError(text_reason_call["error"])
    _assert_evidence_matches_results(text_reason_call, text_results, multimodal=False)

    with Image.open(image_path) as source:
        query_image = source.convert("RGB")
        started = time.perf_counter()
        image_results = service.search_image(query_image, top_k=TOP_K)
        image_latency_ms = (time.perf_counter() - started) * 1000
        assert len(image_results) == TOP_K
        assert len(llm.reason_calls) == 1, "Image-only không được gọi LLM reason"

        started = time.perf_counter()
        multimodal_results, metadata = service.search_multimodal(
            MULTIMODAL_QUERY, query_image, top_k=TOP_K
        )
        multimodal_latency_ms = (time.perf_counter() - started) * 1000

    assert len(multimodal_results) == TOP_K
    assert len(llm.reason_calls) == 2
    multimodal_reason_call = copy.deepcopy(llm.reason_calls[1])
    if "error" in multimodal_reason_call:
        print(json.dumps(multimodal_reason_call, ensure_ascii=False, indent=2))
        raise AssertionError(multimodal_reason_call["error"])
    _assert_evidence_matches_results(
        multimodal_reason_call, multimodal_results, multimodal=True
    )
    assert metadata["llm_status"]["ready"] is True
    assert metadata["llm_status"]["mode"] == "ollama"
    assert "llm_reason_error" not in metadata
    assert metadata["parser"]["parser_mode"] == "ollama"
    assert metadata["alpha"] == SETTINGS.fusion_alpha == 0.8
    assert multimodal_results["price_vnd"].le(1_000_000).all()
    for column in ("text_score", "image_score", "final_score"):
        assert column in multimodal_results
        assert multimodal_results[column].map(math.isfinite).all()

    text_traceability = _traceability(text_reason_call)
    multimodal_traceability = _traceability(multimodal_reason_call)
    assert text_traceability["product_ids_match"]
    assert multimodal_traceability["product_ids_match"]
    if not all(
        not item["ungrounded_numeric_claims"]
        for trace in (text_traceability, multimodal_traceability)
        for item in trace["items"]
    ):
        print(
            json.dumps(
                {
                    "text_traceability": text_traceability,
                    "multimodal_traceability": multimodal_traceability,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    assert all(
        not item["ungrounded_numeric_claims"]
        for trace in (text_traceability, multimodal_traceability)
        for item in trace["items"]
    )
    assert any(
        item["has_real_review_evidence"]
        for trace in (text_traceability, multimodal_traceability)
        for item in trace["items"]
    )

    report = {
        "generated_by": "scripts/verify_phase7_integration.py",
        "verification_phase": PHASE_LABEL,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ollama_status": {
            "ready": status.ready,
            "mode": status.mode,
            "detail": status.detail,
            "model": llm.model,
        },
        "settings": {"fusion_alpha": SETTINGS.fusion_alpha},
        "image_query_product": {
            "product_id": str(image_product["product_id"]),
            "product_name": str(image_product["product_name"]),
            "price_vnd": int(image_product["price_vnd"]),
            "image_path": str(image_product["image_path"]),
        },
        "text_only": {
            "query": TEXT_QUERY,
            "latency_ms": round(text_latency_ms, 3),
            "results": _frame_records(text_results),
            "llm_call": text_reason_call,
            "traceability": text_traceability,
        },
        "image_only": {
            "latency_ms": round(image_latency_ms, 3),
            "results": _frame_records(image_results),
            "llm_reason_call_count_after_mode": 1,
        },
        "multimodal": {
            "query": MULTIMODAL_QUERY,
            "latency_ms": round(multimodal_latency_ms, 3),
            "metadata": metadata,
            "price_filter_max_vnd": 1_000_000,
            "all_results_within_price_filter": bool(
                multimodal_results["price_vnd"].le(1_000_000).all()
            ),
            "results": _frame_records(multimodal_results),
            "llm_call": multimodal_reason_call,
            "traceability": multimodal_traceability,
        },
        "checks": {
            "text_encoder_ran": True,
            "image_encoder_ran": True,
            "fusion_ran": True,
            "price_filter_ran": True,
            "llm_explanations_ran": True,
            "alpha_unchanged": SETTINGS.fusion_alpha == 0.8,
            "numeric_claims_grounded": True,
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"WROTE {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
