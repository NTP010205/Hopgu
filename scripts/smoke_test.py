from __future__ import annotations

import json
import sys
from pathlib import Path
from time import perf_counter

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EVAL_RESULTS_DIR, PROJECT_ROOT
from src.data.loader import load_products, load_reviews
from src.evaluation.metrics import load_evaluation_results
from src.recommender.service import RecommendationService


def timed(callable_):
    start = perf_counter()
    value = callable_()
    return value, (perf_counter() - start) * 1000.0


def main() -> int:
    products = load_products()
    reviews = load_reviews()
    if len(products) < 800 or products["category"].nunique() < 5:
        raise AssertionError("Catalog demo chưa đạt 800 sản phẩm và tối thiểu 5 danh mục")

    target_rows = products.loc[
        products["product_name"].str.contains("WF-1000XM5", case=False, na=False)
    ]
    target = target_rows.iloc[0] if not target_rows.empty else products.iloc[0]
    with Image.open(PROJECT_ROOT / str(target["image_path"])) as source:
        query_image = source.convert("RGB")

    service = RecommendationService(products=products)
    query = str(target["product_name"])

    # Warmup thật: nạp cả hai model và thực hiện inference, không mock dependency.
    service.search_text(query, top_k=3)
    service.search_image(query_image, top_k=3)
    service.search_multimodal(query, query_image, top_k=3)

    text_result, text_ms = timed(lambda: service.search_text(query, top_k=5))
    image_result, image_ms = timed(lambda: service.search_image(query_image, top_k=5))
    (multimodal_result, metadata), multimodal_ms = timed(
        lambda: service.search_multimodal(query, query_image, top_k=5)
    )

    target_id = str(target["product_id"])
    if str(image_result.iloc[0]["product_id"]) != target_id:
        raise AssertionError("Image self-query không trả đúng sản phẩm ở top 1")
    if str(multimodal_result.iloc[0]["product_id"]) != target_id:
        raise AssertionError("Multimodal self-query không trả đúng sản phẩm ở top 1")
    if text_result.empty:
        raise AssertionError("Text search không trả kết quả")
    if not (multimodal_result["text_score"].notna().all() and multimodal_result["image_score"].notna().all()):
        raise AssertionError("Multimodal thiếu score thành phần")

    evaluation = load_evaluation_results(EVAL_RESULTS_DIR / "latest.json")
    if evaluation is None:
        raise AssertionError("Chưa có kết quả evaluation thật")

    report = {
        "ok": True,
        "python": sys.version.split()[0],
        "dataset": {
            "products": len(products),
            "reviews": len(reviews),
            "categories": int(products["category"].nunique()),
        },
        "search": {
            "text_warm_ms": text_ms,
            "image_warm_ms": image_ms,
            "multimodal_warm_ms": multimodal_ms,
            "target_product_id": target_id,
        },
        "llm": metadata["llm_status"],
        "parser_mode": metadata["parser"]["parser_mode"],
        "evaluation_generated_at": evaluation["generated_at"],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if text_ms >= 3000:
        print("WARNING: text warm latency vượt mục tiêu 3 giây", file=sys.stderr)
    if image_ms >= 5000 or multimodal_ms >= 5000:
        print("WARNING: image/multimodal warm latency vượt mục tiêu 5 giây", file=sys.stderr)
    if not metadata["llm_status"]["ready"]:
        print("WARNING: Ollama chưa sẵn sàng; fallback đã được kiểm chứng", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

