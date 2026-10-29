from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EVAL_DIR, EVAL_RESULTS_DIR
from src.data.loader import load_products
from src.recommender.service import RecommendationService


def summarize(values: list[float]) -> dict:
    return {
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def main() -> int:
    manifest_path = EVAL_DIR / "out_of_catalog_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    products = load_products().reset_index(drop=True)
    catalog_ids = set(products["product_id"].astype(str))
    heldout_ids = {str(item["product_id"]) for item in manifest["items"]}
    overlap = sorted(catalog_ids & heldout_ids)
    if overlap:
        raise AssertionError(f"Held-out bị lẫn vào catalog: {overlap}")

    service = RecommendationService(products=products)
    rows: list[dict] = []
    for item in manifest["items"]:
        image_path = PROJECT_ROOT / item["image_path"]
        start = perf_counter()
        with Image.open(image_path) as source:
            results = service.search_image(source.convert("RGB"), top_k=20)
        latency_ms = (perf_counter() - start) * 1000.0
        expected = str(item["expected_category"])
        top20_categories = results["category"].astype(str).tolist()
        category_precision_20 = float(
            np.mean([category == expected for category in top20_categories])
        )
        top5_consistency = float(
            np.mean([category == expected for category in top20_categories[:5]])
        )
        metadata = results.attrs["search_metadata"]
        rows.append(
            {
                "group": item["group"],
                "heldout_product_id": str(item["product_id"]),
                "heldout_product_name": item["product_name"],
                "expected_category": expected,
                "image_path": item["image_path"],
                "image_sha256": item["image_sha256"],
                "predicted_category": metadata["category_inference"]["primary_category"],
                "category_inference": metadata["category_inference"],
                "candidate_count": metadata["candidate_count"],
                "category_precision_at_20": category_precision_20,
                "top5_category_consistency": top5_consistency,
                "latency_ms": latency_ms,
                "top5": [
                    {
                        "rank": rank,
                        "product_id": str(row.product_id),
                        "product_name": str(row.product_name),
                        "category": str(row.category),
                        "image_score": float(row.image_score),
                        "final_score": float(row.final_score),
                    }
                    for rank, row in enumerate(
                        results.head(5).itertuples(index=False), start=1
                    )
                ],
            }
        )

    by_group = {}
    for group in sorted({row["group"] for row in rows}):
        group_rows = [row for row in rows if row["group"] == group]
        by_group[group] = {
            "images": len(group_rows),
            "category_precision_at_20": float(
                np.mean([row["category_precision_at_20"] for row in group_rows])
            ),
            "top5_category_consistency": float(
                np.mean([row["top5_category_consistency"] for row in group_rows])
            ),
        }

    latest = json.loads((EVAL_RESULTS_DIR / "latest.json").read_text(encoding="utf-8"))
    self_image = latest["manual_test"]["systems"]["image_only"]
    output_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "heldout_source": manifest["source"],
            "usage": manifest["usage"],
            "catalog_overlap_count": len(overlap),
            "note": "Ảnh held-out không thuộc catalog và không được dùng để train/tune.",
        },
        "in_catalog_self_image": {
            "label": "IN-CATALOG SELF IMAGE",
            "query_count": latest["manual_test"]["query_count"],
            "metrics": self_image,
        },
        "out_of_catalog_image": {
            "label": "OUT-OF-CATALOG IMAGE",
            "query_count": len(rows),
            "category_precision_at_20": summarize(
                [row["category_precision_at_20"] for row in rows]
            ),
            "top5_category_consistency": summarize(
                [row["top5_category_consistency"] for row in rows]
            ),
            "latency_ms": summarize([row["latency_ms"] for row in rows]),
            "by_group": by_group,
            "cases": rows,
        },
    }
    output = PROJECT_ROOT / "data" / "verification" / "phase9_out_of_catalog.json"
    output.write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(output),
        "catalog_overlap_count": len(overlap),
        "out_of_catalog": {
            "images": len(rows),
            "category_precision_at_20": output_payload["out_of_catalog_image"]["category_precision_at_20"],
            "top5_category_consistency": output_payload["out_of_catalog_image"]["top5_category_consistency"],
            "by_group": by_group,
        },
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
