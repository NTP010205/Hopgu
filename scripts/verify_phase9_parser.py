from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_products
from src.nlp.llm_service import OllamaService
from src.nlp.query_parser import QueryParser


CASES = (
    ("túi xách công sở", "Thời trang nữ", "high"),
    ("túi đeo chéo màu đen", "Thời trang nữ", "high"),
    ("balo laptop chống nước", "Điện tử & Phụ kiện", "high"),
    ("giày sneaker màu trắng", "Thời trang nam", "medium"),
    ("sneaker nữ đế thấp", "Thời trang nữ", "high"),
    ("tai nghe bluetooth chống ồn", "Điện tử & Phụ kiện", "high"),
    ("headphone không dây", "Điện tử & Phụ kiện", "high"),
    ("kem chống nắng cho da dầu", "Làm đẹp & Sức khỏe", "high"),
    ("mỹ phẩm dưỡng da", "Làm đẹp & Sức khỏe", "high"),
    ("bao cao su siêu mỏng nhạy cảm", "Làm đẹp & Sức khỏe", "high"),
    ("máy pha cà phê espresso gia đình", "Điện gia dụng", "high"),
)


class CapturingOllama(OllamaService):
    def __init__(self) -> None:
        super().__init__(timeout=120.0)
        self.last_raw: dict[str, Any] | None = None

    def parse_query(self, query: str, allowed_categories: list[str]) -> dict[str, Any]:
        self.last_raw = super().parse_query(query, allowed_categories)
        return self.last_raw


def main() -> int:
    products = load_products()
    categories = sorted(products["category"].dropna().astype(str).unique())
    llm = CapturingOllama()
    status = llm.status()
    if not status.ready:
        raise RuntimeError(f"Ollama chưa READY: {status.detail}")
    parser = QueryParser(llm, allowed_categories=list(categories))
    rows = []
    for query, expected_category, expected_confidence in CASES:
        start = perf_counter()
        parsed = parser.parse(query)
        latency_ms = (perf_counter() - start) * 1000.0
        if parsed.parser_mode != "ollama":
            raise AssertionError(f"Không đi qua Ollama: {query}")
        if parsed.category != expected_category:
            raise AssertionError(
                f"Sai category {query}: {parsed.category!r} != {expected_category!r}"
            )
        if parsed.category_confidence != expected_confidence:
            raise AssertionError(
                f"Sai confidence {query}: {parsed.category_confidence!r}"
            )
        rows.append({
            "query": query,
            "expected_category": expected_category,
            "expected_confidence": expected_confidence,
            "latency_ms": latency_ms,
            "raw_llm_json": llm.last_raw,
            "validated_parser_output": parsed.to_dict(),
        })
        print(json.dumps(rows[-1], ensure_ascii=False))
    payload = {
        "generated_by": "scripts/verify_phase9_parser.py",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "catalog_products": len(products),
        "allowed_categories": list(categories),
        "ollama_status": {
            "ready": status.ready,
            "mode": status.mode,
            "detail": status.detail,
            "model": llm.model,
        },
        "query_count": len(rows),
        "queries": rows,
    }
    output = PROJECT_ROOT / "data" / "verification" / "phase9_parser_outputs.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"WROTE {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
