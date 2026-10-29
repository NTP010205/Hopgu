from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SETTINGS
from src.nlp.review_analyzer import top_review_evidence
from src.recommender.filters import apply_filters
from src.recommender.fusion import fuse_scores
from src.recommender.scoring import cosine_scores
from src.recommender.service import RecommendationService


def timed(callable_):
    start = perf_counter()
    result = callable_()
    return result, (perf_counter() - start) * 1000.0


def main() -> int:
    service = RecommendationService()
    target = service.products.loc[
        service.products["product_name"].str.contains("WF-1000XM5", case=False, na=False)
    ].iloc[0]
    query = "tai nghe bluetooth true wireless chống ồn dưới 5 triệu"
    with Image.open(PROJECT_ROOT / str(target["image_path"])) as source:
        image = source.convert("RGB")

    # Warmup model loading is deliberately outside steady-state stage timings.
    service.text_encoder.encode_query("khởi động")
    service.image_encoder.encode_image(image)
    service.load_text_matrix()
    service.load_image_matrix()

    parsed, parser_ms = timed(lambda: service.query_parser.parse(query))
    text_vector, text_embedding_ms = timed(
        lambda: service.text_encoder.encode_query(query)
    )
    image_vector, image_embedding_ms = timed(
        lambda: service.image_encoder.encode_image(image)
    )

    def retrieve():
        hard, soft, policy = service._text_category_policy(parsed)
        if hard:
            candidate_categories = hard
            inference = policy
        elif soft:
            candidate_categories = ()
            inference = policy
        else:
            candidate_categories, inference = service._image_category_candidates(
                image_vector
            )
        eligible = apply_filters(
            service.products,
            categories=candidate_categories or None,
            price_min=parsed.price_min,
            price_max=parsed.price_max,
        )
        indices = eligible.index.to_numpy(dtype=int)
        text_scores = cosine_scores(service.text_matrix, text_vector)
        image_scores = cosine_scores(service.image_matrix, image_vector)
        fused = fuse_scores(
            text_scores[indices], image_scores[indices], alpha=SETTINGS.fusion_alpha
        )
        fused += service._soft_category_bonus(indices, soft)
        order = np.argsort(fused)[::-1][:5]
        ranked = service.products.loc[indices[order]].copy()
        ranked["text_score"] = text_scores[indices[order]]
        ranked["image_score"] = image_scores[indices[order]]
        ranked["final_score"] = fused[order]
        return ranked, len(eligible), inference

    (ranked, candidate_count, inference), retrieval_ms = timed(retrieve)
    evidence = [
        {
            "product_id": str(row.product_id),
            "name": str(row.product_name),
            "category": str(row.category),
            "price_vnd": int(row.price_vnd),
            "rating": float(row.rating),
            "description": str(row.description).strip()
            if pd.notna(row.description)
            else "",
            "review": top_review_evidence(row.reviews_text),
            "text_score": float(row.text_score),
            "image_score": float(row.image_score),
            "recommendation_score": float(row.final_score),
        }
        for row in ranked.itertuples(index=False)
    ]
    status = service.llm.status()
    reason_error = None
    reasons = {}
    explanation_ms = 0.0
    parser_cache_hit_ms = None
    explanation_cache_hit_ms = None
    if status.ready:
        try:
            reasons, explanation_ms = timed(
                lambda: service.llm.generate_reasons(query, evidence)
            )
            _, parser_cache_hit_ms = timed(lambda: service.query_parser.parse(query))
            _, explanation_cache_hit_ms = timed(
                lambda: service.llm.generate_reasons(query, evidence)
            )
        except Exception as exc:
            reason_error = f"{type(exc).__name__}: {exc}"

    stages = {
        "parser_llm_ms": parser_ms,
        "text_embedding_ms": text_embedding_ms,
        "image_embedding_ms": image_embedding_ms,
        "retrieval_ranking_ms": retrieval_ms,
        "explanation_generation_top5_ms": explanation_ms,
    }
    total = sum(stages.values())
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "single warmed multimodal request; real local models and Ollama; top-5 explanations",
        "query": query,
        "query_image_product_id": str(target["product_id"]),
        "llm_status": {
            "ready": status.ready,
            "mode": status.mode,
            "detail": status.detail,
        },
        "parsed_query": parsed.to_dict(),
        "category_policy": inference,
        "candidate_count": candidate_count,
        "stages_ms": stages,
        "measured_total_ms": total,
        "stage_share": {
            key: (value / total if total else 0.0) for key, value in stages.items()
        },
        "cache_hit_ms": {
            "parser_llm": parser_cache_hit_ms,
            "explanation_generation_top5": explanation_cache_hit_ms,
        },
        "reason_count": len(reasons),
        "reason_error": reason_error,
        "top5_product_ids": ranked["product_id"].astype(str).tolist(),
    }
    output = PROJECT_ROOT / "data" / "verification" / "phase9_latency_profile.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), **payload}, ensure_ascii=False, indent=2))
    return 0 if not reason_error else 1


if __name__ == "__main__":
    raise SystemExit(main())
