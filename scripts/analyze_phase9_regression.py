from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate import Evaluator, load_manual_queries
from src.config import EVAL_DIR
from src.evaluation.metrics import hit_at_k, ndcg_at_k, precision_at_k
from src.recommender.filters import apply_filters
from src.recommender.fusion import fuse_scores
from src.recommender.scoring import cosine_scores
from src.recommender.service import CATEGORY_SOFT_BONUS


def main() -> int:
    output = PROJECT_ROOT / "data" / "verification" / "phase9_multimodal_regression.json"
    payload = json.loads(output.read_text(encoding="utf-8"))
    evaluator = Evaluator()
    cases = [
        case
        for case in load_manual_queries(EVAL_DIR / "queries.jsonl")
        if case.split == "test"
    ]
    old_by_case = {row["case_id"]: row for row in payload["cases"]}
    phase9_ndcg: list[float] = []

    for case in cases:
        parsed = evaluator.query_parser.parse(case.query)
        text_vector = evaluator.text_encoder.encode_query(case.query)
        text_scores = cosine_scores(evaluator.text_matrix, text_vector)
        image_index = evaluator.id_to_index[case.image_product_id]
        image_path = PROJECT_ROOT / str(evaluator.products.iloc[image_index]["image_path"])
        with Image.open(image_path) as source:
            image_vector = evaluator.image_encoder.encode_image(source.convert("RGB"))
        image_scores = cosine_scores(evaluator.image_matrix, image_vector)
        image_prediction = evaluator.category_classifier.predict_vector(image_vector)

        hard_categories: tuple[str, ...] = ()
        soft_categories: tuple[str, ...] = ()
        candidate_source = "clip_zero_shot"
        if parsed.category_confidence == "high" and parsed.category:
            hard_categories = tuple(parsed.category_candidates) or (parsed.category,)
            candidate_source = "text_query_high_confidence"
        elif parsed.category_confidence == "medium" and parsed.category_candidates:
            soft_categories = tuple(parsed.category_candidates)
            candidate_source = "text_query_soft_rerank"
        else:
            hard_categories = image_prediction.candidate_categories

        eligible = apply_filters(
            evaluator.products,
            categories=hard_categories or None,
            price_min=parsed.price_min,
            price_max=parsed.price_max,
        )
        indices = eligible.index.to_numpy(dtype=int)
        fused = fuse_scores(text_scores[indices], image_scores[indices], alpha=0.8)
        if soft_categories:
            bonus = evaluator.products.loc[indices, "category"].astype(str).isin(
                soft_categories
            ).to_numpy(dtype=np.float32)
            fused = fused + bonus * CATEGORY_SOFT_BONUS
        order = np.argsort(fused)[::-1][:20]
        ranked_indices = indices[order]
        ranked_ids = [evaluator.product_ids[index] for index in ranked_indices]
        relevant = evaluator.relevant_ids(case)
        eligible_ids = set(eligible["product_id"].astype(str))
        relevant_eliminated = sorted(relevant - eligible_ids)
        top20 = []
        for rank, (index, score) in enumerate(zip(ranked_indices, fused[order], strict=True), start=1):
            row = evaluator.products.iloc[index]
            top20.append(
                {
                    "rank": rank,
                    "product_id": str(row["product_id"]),
                    "product_name": str(row["product_name"]),
                    "category": str(row["category"]),
                    "relevant": str(row["product_id"]) in relevant,
                    "text_score": float(text_scores[index]),
                    "image_score": float(image_scores[index]),
                    "category_bonus": (
                        CATEGORY_SOFT_BONUS
                        if soft_categories and str(row["category"]) in soft_categories
                        else 0.0
                    ),
                    "final_score": float(score),
                }
            )
        metrics = {
            "precision_at_5": precision_at_k(ranked_ids[:5], relevant, 5),
            "hit_at_5": hit_at_k(ranked_ids[:5], relevant, 5),
            "ndcg_at_5": ndcg_at_k(ranked_ids[:5], relevant, 5),
            "category_precision_at_20": float(
                evaluator.products.loc[ranked_indices, "category"].astype(str).isin(
                    case.relevant_categories
                ).mean()
            ),
        }
        phase9_ndcg.append(metrics["ndcg_at_5"])
        old = old_by_case[case.case_id]
        previous_cause = old.get("root_causes", ["không gây regression"])
        if metrics["ndcg_at_5"] >= old["phase7"]["metrics"]["ndcg_at_5"] - 1e-12:
            resolution = "resolved_or_not_regressed"
        elif metrics["ndcg_at_5"] > old["phase8"]["metrics"]["ndcg_at_5"]:
            resolution = "improved_but_below_phase7"
        else:
            resolution = "not_resolved"
        old["phase9"] = {
            "parsed_query": parsed.to_dict(),
            "image_inference": image_prediction.to_dict(),
            "candidate_source": candidate_source,
            "hard_candidate_categories": list(hard_categories),
            "soft_candidate_categories": list(soft_categories),
            "candidate_count": len(eligible),
            "relevant_items_eliminated_count": len(relevant_eliminated),
            "relevant_items_eliminated_preview": relevant_eliminated[:50],
            "metrics": metrics,
            "top20": top20,
            "previous_root_causes": previous_cause,
            "resolution": resolution,
        }

    latest = json.loads(
        (PROJECT_ROOT / "data" / "eval" / "results" / "latest.json").read_text(
            encoding="utf-8"
        )
    )
    payload["phase9_generated_at"] = datetime.now(timezone.utc).isoformat()
    payload["phase9_definition"] = (
        "high-confidence text category hard gate; medium-confidence text category "
        "global candidate pool + fixed 0.03 category bonus; otherwise CLIP zero-shot "
        "guardrail; alpha remains 0.8"
    )
    payload["phase9_summary"] = latest["manual_test"]["systems"]["multimodal"]
    payload["phase9_mean_ndcg_recomputed"] = float(np.mean(phase9_ndcg))
    payload["phase9_fix_notes"] = [
        "bao cao su -> Làm đẹp & Sức khỏe, high confidence",
        "máy pha cà phê/espresso -> Điện gia dụng, high confidence",
        "ambiguous giày/sneaker -> global pool with male/female category bonus",
        "explicit category from UI or high-confidence parser remains a hard constraint",
    ]
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(output),
        "cases": len(cases),
        "phase9_ndcg": payload["phase9_mean_ndcg_recomputed"],
        "resolutions": {
            value: sum(
                row["phase9"]["resolution"] == value for row in payload["cases"]
            )
            for value in ("resolved_or_not_regressed", "improved_but_below_phase7", "not_resolved")
        },
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
