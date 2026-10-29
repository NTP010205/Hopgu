from __future__ import annotations

import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EMBEDDINGS_DIR, EVAL_DIR, EVAL_RESULTS_DIR, PROJECT_ROOT, SETTINGS
from src.data.loader import load_products
from src.evaluation.metrics import hit_at_k, ndcg_at_k, precision_at_k, validate_result_payload
from src.nlp.llm_service import LLMStatus
from src.nlp.query_parser import QueryParser
from src.nlp.text_encoder import TextEncoder
from src.recommender.filters import apply_filters
from src.recommender.fusion import fuse_scores
from src.recommender.scoring import cosine_scores
from src.recommender.service import CATEGORY_SOFT_BONUS
from src.vision.category_classifier import ZeroShotCategoryClassifier
from src.vision.image_encoder import ImageEncoder


@dataclass(slots=True)
class EvalCase:
    case_id: str
    query: str
    image_product_id: str
    relevant_product_ids: set[str]
    relevant_categories: set[str]
    split: str


class OfflineEvalLLM:
    def status(self) -> LLMStatus:
        return LLMStatus(False, "fallback", "evaluation deterministic rule parser")


def load_manual_queries(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        case = EvalCase(
            case_id=str(payload["id"]),
            query=str(payload["query"]),
            image_product_id=str(payload["query_image_product_id"]),
            relevant_product_ids={str(value) for value in payload.get("relevant_product_ids", [])},
            relevant_categories={str(value) for value in payload.get("relevant_categories", [])},
            split=str(payload["split"]),
        )
        if case.split not in {"validation", "test"}:
            raise ValueError(f"Dòng {line_number}: split không hợp lệ")
        cases.append(case)
    if len(cases) < 30:
        raise ValueError("Manual query set phải có ít nhất 30 query")
    if not any(case.split == "validation" for case in cases) or not any(case.split == "test" for case in cases):
        raise ValueError("Query set phải tách validation và test")
    return cases


class Evaluator:
    def __init__(self) -> None:
        self.products = load_products().reset_index(drop=True)
        self.product_ids = self.products["product_id"].astype(str).tolist()
        self.id_to_index = {product_id: index for index, product_id in enumerate(self.product_ids)}
        self.text_matrix = np.load(EMBEDDINGS_DIR / "text_embeddings.npy", allow_pickle=False)
        self.image_matrix = np.load(EMBEDDINGS_DIR / "image_embeddings.npy", allow_pickle=False)
        if self.text_matrix.shape[0] != len(self.products) or self.image_matrix.shape[0] != len(self.products):
            raise ValueError("Embedding cache không khớp catalog")
        self.text_encoder = TextEncoder(offline=True)
        self.image_encoder = ImageEncoder(offline=True)
        categories = sorted(self.products["category"].dropna().astype(str).unique().tolist())
        self.query_parser = QueryParser(OfflineEvalLLM(), allowed_categories=categories)
        self.category_classifier = ZeroShotCategoryClassifier(self.image_encoder, categories)

    def relevant_ids(self, case: EvalCase) -> set[str]:
        relevant = set(case.relevant_product_ids)
        if case.relevant_categories:
            matched = self.products["category"].astype(str).isin(case.relevant_categories)
            relevant.update(self.products.loc[matched, "product_id"].astype(str))
        return relevant

    def query_scores(self, case: EvalCase, mode: str, alpha: float) -> tuple[np.ndarray, float]:
        start = perf_counter()
        text_scores = image_scores = None
        parsed = (
            self.query_parser.parse(case.query)
            if mode in {"text_only", "multimodal"}
            else None
        )
        image_vector = None
        if mode in {"text_only", "multimodal"}:
            text_vector = self.text_encoder.encode_query(case.query)
            text_scores = cosine_scores(self.text_matrix, text_vector)
        if mode in {"image_only", "multimodal"}:
            index = self.id_to_index.get(case.image_product_id)
            if index is None:
                raise ValueError(f"Ảnh query product_id không có trong catalog: {case.image_product_id}")
            image_path = PROJECT_ROOT / str(self.products.iloc[index]["image_path"])
            with Image.open(image_path) as source:
                image_vector = self.image_encoder.encode_image(source.convert("RGB"))
            image_scores = cosine_scores(self.image_matrix, image_vector)

        candidate_categories: tuple[str, ...] = ()
        soft_categories: tuple[str, ...] = ()
        if parsed is not None:
            if parsed.category_confidence == "high" and parsed.category:
                candidate_categories = tuple(parsed.category_candidates) or (parsed.category,)
            elif parsed.category_confidence == "medium":
                soft_categories = tuple(parsed.category_candidates)
        if (
            not candidate_categories
            and not soft_categories
            and mode in {"image_only", "multimodal"}
            and image_vector is not None
        ):
            candidate_categories = self.category_classifier.predict_vector(
                image_vector
            ).candidate_categories
        eligible = apply_filters(
            self.products,
            categories=candidate_categories or None,
            price_min=parsed.price_min if parsed is not None else None,
            price_max=parsed.price_max if parsed is not None else None,
        )
        indices = eligible.index.to_numpy(dtype=int)
        if not len(indices):
            return np.full(len(self.products), -np.inf, dtype=np.float32), (
                perf_counter() - start
            ) * 1000.0

        if mode == "text_only":
            candidate_scores = text_scores[indices]
        elif mode == "image_only":
            candidate_scores = image_scores[indices]
        elif mode == "multimodal":
            candidate_scores = fuse_scores(
                text_scores[indices], image_scores[indices], alpha=alpha
            )
        else:
            raise ValueError(f"Mode không hợp lệ: {mode}")
        if soft_categories:
            bonus = self.products.loc[indices, "category"].astype(str).isin(
                soft_categories
            ).to_numpy(dtype=np.float32)
            candidate_scores = candidate_scores + bonus * CATEGORY_SOFT_BONUS
        final = np.full(len(self.products), -np.inf, dtype=np.float32)
        final[indices] = candidate_scores
        latency_ms = (perf_counter() - start) * 1000.0
        return np.asarray(final), latency_ms

    def evaluate(self, cases: list[EvalCase], mode: str, alpha: float) -> tuple[dict, list[dict]]:
        rows: list[dict] = []
        for case in cases:
            scores, latency = self.query_scores(case, mode, alpha)
            ranked_indices = np.argsort(scores)[::-1][:20]
            ranked_ids = [self.product_ids[index] for index in ranked_indices[:5]]
            relevant = self.relevant_ids(case)
            row = {
                "case_id": case.case_id,
                "mode": mode,
                "precision_at_5": precision_at_k(ranked_ids, relevant, 5),
                "hit_at_5": hit_at_k(ranked_ids, relevant, 5),
                "ndcg_at_5": ndcg_at_k(ranked_ids, relevant, 5),
                "latency_ms": latency,
                "top_ids": ranked_ids,
                "category_precision_at_20": None,
            }
            if mode in {"image_only", "multimodal"}:
                top_categories = self.products.iloc[ranked_indices]["category"].astype(str)
                row["category_precision_at_20"] = float(
                    top_categories.isin(case.relevant_categories).mean()
                )
            rows.append(row)
        summary = {
            "precision_at_5": float(np.mean([row["precision_at_5"] for row in rows])),
            "hit_at_5": float(np.mean([row["hit_at_5"] for row in rows])),
            "ndcg_at_5": float(np.mean([row["ndcg_at_5"] for row in rows])),
            "latency_ms_mean": float(np.mean([row["latency_ms"] for row in rows])),
        }
        if mode in {"image_only", "multimodal"}:
            summary["category_precision_at_20"] = float(
                np.mean([row["category_precision_at_20"] for row in rows])
            )
        return summary, rows

    def proxy_cases(self, count: int = 32) -> list[EvalCase]:
        indices = np.linspace(0, len(self.products) - 1, count, dtype=int)
        cases: list[EvalCase] = []
        for number, index in enumerate(indices, start=1):
            row = self.products.iloc[index]
            description = str(row["description"])[:240]
            cases.append(
                EvalCase(
                    case_id=f"proxy_{number:03d}",
                    query=f"{row['product_name']}. {description}",
                    image_product_id=str(row["product_id"]),
                    relevant_product_ids={str(row["product_id"])},
                    relevant_categories={str(row["category"])},
                    split="proxy",
                )
            )
        return cases


def ids_sha256(product_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(product_ids).encode("utf-8")).hexdigest()


def main() -> int:
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    evaluator = Evaluator()
    manual = load_manual_queries(EVAL_DIR / "queries.jsonl")
    validation = [case for case in manual if case.split == "validation"]
    test = [case for case in manual if case.split == "test"]
    alpha = SETTINGS.fusion_alpha
    print(f"ALPHA_FIXED={alpha:.1f}; Phase 9 không retune", flush=True)

    all_rows: list[dict] = []
    manual_systems: dict[str, dict] = {}
    for mode in ("text_only", "image_only", "multimodal"):
        summary, rows = evaluator.evaluate(test, mode, alpha)
        manual_systems[mode] = summary
        all_rows.extend({**row, "scope": "manual_test"} for row in rows)
        print(f"MANUAL {mode}: {summary}", flush=True)

    proxy = evaluator.proxy_cases()
    proxy_systems: dict[str, dict] = {}
    for mode in ("text_only", "image_only", "multimodal"):
        summary, rows = evaluator.evaluate(proxy, mode, alpha)
        proxy_systems[mode] = summary
        all_rows.extend({**row, "scope": "proxy"} for row in rows)
        print(f"PROXY {mode}: {summary}", flush=True)

    text_manifest = json.loads((EMBEDDINGS_DIR / "text_manifest.json").read_text(encoding="utf-8"))
    image_manifest = json.loads((EMBEDDINGS_DIR / "image_manifest.json").read_text(encoding="utf-8"))
    payload = {
        "generated_by": "scripts/evaluate.py",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "k": 5,
        "dataset": {
            "products": len(evaluator.products),
            "product_ids_sha256": ids_sha256(evaluator.product_ids),
            "text_dataset_sha256": text_manifest["dataset_text_sha256"],
            "image_dataset_sha256": image_manifest["dataset_image_sha256"],
        },
        "models": {"text": SETTINGS.text_model, "vision": SETTINGS.vision_model},
        "alpha_selection": {
            "selected": alpha,
            "selected_on": "fixed_from_previous_phase_no_retune",
            "validation_query_count": len(validation),
            "grid": [],
        },
        "evaluation_protocol": {
            "labels_unchanged": True,
            "queries_sha256": hashlib.sha256(
                (EVAL_DIR / "queries.jsonl").read_bytes()
            ).hexdigest(),
            "image_only_uses_text_query": False,
            "image_only_note": (
                "Phase 9 sửa bug evaluator Phase 8: image-only nay chỉ dùng ảnh và "
                "CLIP zero-shot category guardrail, khớp production search_image."
            ),
            "multimodal_category_policy": (
                "text high-confidence hard gate; text medium-confidence global + "
                "fixed category bonus; otherwise CLIP zero-shot guardrail"
            ),
        },
        "manual_test": {"query_count": len(test), "systems": manual_systems},
        "proxy": {
            "query_count": len(proxy),
            "note": "Proxy: query lấy từ title/description/image; cùng broad category được xem là relevant.",
            "systems": proxy_systems,
        },
    }
    baseline_path = PROJECT_ROOT / "data" / "verification" / "phase8_baseline.json"
    if baseline_path.is_file():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))[
            "category_precision_at_20"
        ]
        image_after = manual_systems["image_only"]["category_precision_at_20"]
        multi_after = manual_systems["multimodal"]["category_precision_at_20"]
        payload["category_precision_at_20_comparison"] = {
            "query_count": len(test),
            "before": {
                "image_only": baseline["image_only_mean"],
                "multimodal": baseline["multimodal_mean"],
            },
            "after": {
                "image_only": image_after,
                "multimodal": multi_after,
            },
            "delta": {
                "image_only": image_after - baseline["image_only_mean"],
                "multimodal": multi_after - baseline["multimodal_mean"],
            },
        }
    archive_dir = EVAL_RESULTS_DIR / "archive"
    phase7_path = archive_dir / "phase7_latest_20261011T060127Z.json"
    phase8_path = archive_dir / "phase8_latest_20261011T180749Z.json"
    if phase7_path.is_file() and phase8_path.is_file():
        phase7 = json.loads(phase7_path.read_text(encoding="utf-8"))
        phase8 = json.loads(phase8_path.read_text(encoding="utf-8"))
        payload["phase_comparison"] = {
            "phase7": phase7["manual_test"]["systems"],
            "phase8": phase8["manual_test"]["systems"],
            "phase9": manual_systems,
            "comparability_note": (
                "Labels và alpha giữ nguyên. Phase 8 image-only từng vô tình dùng parsed "
                "text category; Phase 9 sửa bug đó, nên image-only Phase 8/9 không hoàn "
                "toàn cùng protocol. Multimodal vẫn so sánh trực tiếp được."
            ),
        }
    validate_result_payload(payload)
    (EVAL_RESULTS_DIR / "latest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (EVAL_RESULTS_DIR / "latest_details.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["scope", "case_id", "mode", "precision_at_5", "hit_at_5", "ndcg_at_5", "category_precision_at_20", "latency_ms", "top_ids"],
        )
        writer.writeheader()
        for row in all_rows:
            writer.writerow({**row, "top_ids": "|".join(row["top_ids"])})
    print(f"WROTE {EVAL_RESULTS_DIR / 'latest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

