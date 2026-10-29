from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_products
from src.nlp.llm_service import OllamaService
from src.nlp.query_parser import QueryParser


OUTPUT_PATH = Path("data/verification/phase7_parser_outputs.json")
QUERIES = [
    "túi đeo chéo nữ màu đen dưới 500 nghìn",
    "giày chạy bộ nam nhẹ dưới 1 triệu",
    "tai nghe bluetooth chống ồn",
    "kem chống nắng cho da dầu",
    "balo laptop chống nước",
    "tôi muốn một chiếc túi giống hình này nhưng rẻ hơn 600 nghìn",
]
REQUIRED_FIELDS = {
    "intent",
    "category",
    "gender",
    "colors",
    "attributes",
    "price_min",
    "price_max",
    "keywords",
}


class CapturingOllamaService(OllamaService):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.last_raw_parser_output: dict[str, Any] | None = None

    def parse_query(self, query: str, allowed_categories: list[str]) -> dict[str, Any]:
        payload = super().parse_query(query, allowed_categories)
        self.last_raw_parser_output = payload
        return payload


def main() -> None:
    products = load_products()
    allowed_categories = sorted(
        products["category"].dropna().astype(str).unique().tolist()
    )
    llm = CapturingOllamaService(timeout=120.0)
    status = llm.status()
    if not status.ready:
        raise RuntimeError(f"Ollama chưa READY: {status.detail}")

    version = requests.get(f"{llm.base_url}/api/version", timeout=5).json()
    tags = requests.get(f"{llm.base_url}/api/tags", timeout=5).json()
    parser = QueryParser(llm, allowed_categories=allowed_categories)
    records: list[dict[str, Any]] = []

    for query in QUERIES:
        llm.last_raw_parser_output = None
        started = time.perf_counter()
        parsed = parser.parse(query)
        elapsed_ms = (time.perf_counter() - started) * 1000
        parsed_payload = parsed.to_dict()
        raw_payload = llm.last_raw_parser_output
        fallback_payload = parser._rule_parse(
            query, "Phase 7 comparison: rule-based"
        ).to_dict()

        if raw_payload is None:
            raise AssertionError(f"Query không đi qua Ollama: {query}")
        if not isinstance(raw_payload, dict):
            raise AssertionError(f"Raw output không phải JSON object: {query}")
        missing = sorted(REQUIRED_FIELDS - set(parsed_payload))
        if missing:
            raise AssertionError(f"Thiếu field {missing}: {query}")
        if parsed_payload["parser_mode"] != "ollama":
            raise AssertionError(f"Parser không ở mode ollama: {query}")

        records.append(
            {
                "query": query,
                "latency_ms": round(elapsed_ms, 3),
                "raw_llm_json": raw_payload,
                "validated_parser_output": parsed_payload,
                "fallback_parser_output": fallback_payload,
            }
        )
        print(json.dumps(records[-1], ensure_ascii=False, indent=2))

    report = {
        "generated_by": "scripts/verify_phase7_parser.py",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ollama": {
            "base_url": llm.base_url,
            "version": version.get("version"),
            "configured_model": llm.model,
            "status": {
                "ready": status.ready,
                "mode": status.mode,
                "detail": status.detail,
            },
            "models": tags.get("models", []),
        },
        "catalog": {
            "product_count": len(products),
            "allowed_categories": allowed_categories,
        },
        "query_count": len(records),
        "queries": records,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"WROTE {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
