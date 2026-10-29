from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from streamlit.testing.v1 import AppTest

from src.data.loader import load_products
from src.nlp.llm_service import OllamaService
from src.nlp.query_parser import QueryParser
from src.recommender.service import RecommendationService


PHASE_LABEL = os.environ.get("HOPGU_VERIFICATION_PHASE", "phase7")
OUTPUT_PATH = PROJECT_ROOT / f"data/verification/{PHASE_LABEL}_fallback.json"
QUERY = "tai nghe bluetooth chống ồn dưới 1 triệu"


def main() -> None:
    llm = OllamaService(timeout=3.0)
    status = llm.status()
    assert status.ready is False
    assert status.mode == "fallback"

    products = load_products()
    categories = sorted(products["category"].dropna().astype(str).unique().tolist())
    parsed = QueryParser(llm, allowed_categories=categories).parse(QUERY)
    assert parsed.parser_mode == "fallback"
    assert parsed.price_max == 1_000_000

    service = RecommendationService(products=products, llm=llm)
    results = service.search_text(QUERY, top_k=3)
    assert len(results) == 3
    assert set(results["parser_mode"]) == {"fallback"}
    assert results["price_vnd"].le(1_000_000).all()

    app_test = AppTest.from_file(
        str(PROJECT_ROOT / "pages/4_He_thong.py")
    ).run(timeout=30)
    assert not app_test.exception
    warnings = [item.value for item in app_test.warning]
    assert any("FALLBACK" in value for value in warnings)

    report = {
        "generated_by": "scripts/verify_phase7_fallback.py",
        "verification_phase": PHASE_LABEL,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ollama_status": {
            "ready": status.ready,
            "mode": status.mode,
            "detail": status.detail,
        },
        "query": QUERY,
        "parsed_query": parsed.to_dict(),
        "text_search": {
            "result_count": len(results),
            "all_prices_at_most_1m": bool(results["price_vnd"].le(1_000_000).all()),
            "results": [
                {
                    "product_id": str(row.product_id),
                    "product_name": str(row.product_name),
                    "price_vnd": int(row.price_vnd),
                    "parser_mode": str(row.parser_mode),
                }
                for row in results.itertuples(index=False)
            ],
        },
        "system_status_app_test": {
            "exceptions": len(app_test.exception),
            "warnings": warnings,
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
