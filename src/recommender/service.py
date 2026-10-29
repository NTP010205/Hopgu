from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import EMBEDDINGS_DIR
from src.config import SETTINGS
from src.data.loader import load_products
from src.nlp.text_encoder import TextEncoder
from src.nlp.llm_service import OllamaService
from src.nlp.query_parser import QueryParser
from src.nlp.review_analyzer import top_review_evidence
from src.recommender.filters import apply_filters
from src.recommender.fusion import fuse_scores
from src.recommender.scoring import cosine_scores
from src.vision.category_classifier import ZeroShotCategoryClassifier
from src.vision.image_encoder import ImageEncoder


class CacheMismatchError(RuntimeError):
    pass


CATEGORY_SOFT_BONUS = 0.03


class RecommendationService:
    def __init__(
        self,
        *,
        products: pd.DataFrame | None = None,
        text_encoder: TextEncoder | None = None,
        image_encoder: ImageEncoder | None = None,
        llm: OllamaService | None = None,
        embeddings_dir: Path = EMBEDDINGS_DIR,
    ) -> None:
        self.products = products if products is not None else load_products()
        self.products = self.products.reset_index(drop=True)
        self.text_encoder = text_encoder or TextEncoder()
        self.image_encoder = image_encoder or ImageEncoder()
        self.llm = llm or OllamaService()
        self.query_parser = QueryParser(
            self.llm,
            allowed_categories=sorted(self.products["category"].dropna().astype(str).unique().tolist()),
        )
        self.category_classifier = ZeroShotCategoryClassifier(
            self.image_encoder,
            self.query_parser.allowed_categories,
        )
        self.embeddings_dir = embeddings_dir
        self._text_matrix: np.ndarray | None = None
        self._image_matrix: np.ndarray | None = None

    @staticmethod
    def _selected_category(category: str | None) -> str | None:
        return category if category and category != "Tất cả" else None

    @staticmethod
    def _effective_price_bounds(
        price_min: int | None,
        price_max: int | None,
        parsed_min: int | None = None,
        parsed_max: int | None = None,
    ) -> tuple[int | None, int | None]:
        effective_min = max(
            (value for value in (price_min, parsed_min) if value is not None),
            default=None,
        )
        effective_max = min(
            (value for value in (price_max, parsed_max) if value is not None),
            default=None,
        )
        if (
            effective_min is not None
            and effective_max is not None
            and effective_min > effective_max
        ):
            raise ValueError("Khoảng giá trong truy vấn xung đột với bộ lọc đã chọn")
        return effective_min, effective_max

    @staticmethod
    def _text_category_policy(parsed, explicit_category: str | None = None) -> tuple[tuple[str, ...], tuple[str, ...], dict]:
        if explicit_category:
            return (explicit_category,), (), {
                "source": "ui_filter",
                "confidence": "explicit",
                "strategy": "hard_constraint",
            }
        candidates = tuple(parsed.category_candidates)
        if parsed.category_confidence == "high" and parsed.category:
            hard = candidates or (parsed.category,)
            return hard, (), {
                "source": "text_query",
                "confidence": "high",
                "strategy": "hard_constraint",
                "alias": parsed.category_alias,
            }
        if parsed.category_confidence == "medium" and candidates:
            return (), candidates, {
                "source": "text_query",
                "confidence": "medium",
                "strategy": "global_with_category_bonus",
                "alias": parsed.category_alias,
            }
        return (), (), {
            "source": "text_query",
            "confidence": "none",
            "strategy": "global_catalog",
            "alias": parsed.category_alias,
        }

    def _soft_category_bonus(
        self,
        indices: np.ndarray,
        categories: tuple[str, ...],
    ) -> np.ndarray:
        if not categories:
            return np.zeros(len(indices), dtype=np.float32)
        matched = self.products.loc[indices, "category"].astype(str).isin(categories)
        return matched.to_numpy(dtype=np.float32) * CATEGORY_SOFT_BONUS

    def _image_category_candidates(
        self,
        image_vector: np.ndarray,
        *,
        explicit_category: str | None = None,
        text_category: str | None = None,
    ) -> tuple[tuple[str, ...], dict]:
        if explicit_category:
            return (explicit_category,), {
                "source": "ui_filter",
                "primary_category": explicit_category,
                "candidate_categories": [explicit_category],
                "confidence": "explicit",
                "uncertain": False,
                "strategy": "hard_constraint",
            }
        if text_category:
            return (text_category,), {
                "source": "text_query",
                "primary_category": text_category,
                "candidate_categories": [text_category],
                "confidence": "explicit",
                "uncertain": False,
                "strategy": "hard_constraint",
            }
        prediction = self.category_classifier.predict_vector(image_vector)
        metadata = prediction.to_dict()
        metadata["source"] = "clip_zero_shot"
        return prediction.candidate_categories, metadata

    def load_text_matrix(self) -> np.ndarray:
        matrix_path = self.embeddings_dir / "text_embeddings.npy"
        manifest_path = self.embeddings_dir / "text_manifest.json"
        if not matrix_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(
                "Chưa có text embeddings. Hãy chạy python scripts/build_text_embeddings.py"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_ids = self.products["product_id"].astype(str).tolist()
        if manifest.get("product_ids") != expected_ids:
            raise CacheMismatchError("Text embedding cache không khớp thứ tự product_id")
        matrix = np.load(matrix_path, allow_pickle=False)
        if matrix.shape[0] != len(self.products):
            raise CacheMismatchError("Số dòng text embedding không khớp catalog")
        self._text_matrix = np.asarray(matrix, dtype=np.float32)
        return self._text_matrix

    @property
    def text_matrix(self) -> np.ndarray:
        return self._text_matrix if self._text_matrix is not None else self.load_text_matrix()

    def load_image_matrix(self) -> np.ndarray:
        matrix_path = self.embeddings_dir / "image_embeddings.npy"
        manifest_path = self.embeddings_dir / "image_manifest.json"
        if not matrix_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(
                "Chưa có image embeddings. Hãy chạy python scripts/build_image_embeddings.py"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_ids = self.products["product_id"].astype(str).tolist()
        if manifest.get("product_ids") != expected_ids:
            raise CacheMismatchError("Image embedding cache không khớp thứ tự product_id")
        matrix = np.load(matrix_path, allow_pickle=False)
        if matrix.shape[0] != len(self.products):
            raise CacheMismatchError("Số dòng image embedding không khớp catalog")
        self._image_matrix = np.asarray(matrix, dtype=np.float32)
        return self._image_matrix

    @property
    def image_matrix(self) -> np.ndarray:
        return self._image_matrix if self._image_matrix is not None else self.load_image_matrix()

    def search_text(
        self,
        query: str,
        *,
        category: str | None = None,
        price_min: int | None = None,
        price_max: int | None = None,
        top_k: int = 20,
    ) -> pd.DataFrame:
        if not query.strip():
            raise ValueError("Vui lòng nhập nội dung tìm kiếm")
        parsed = self.query_parser.parse(query)
        hard_categories, soft_categories, category_policy = self._text_category_policy(
            parsed,
            self._selected_category(category),
        )
        effective_min, effective_max = self._effective_price_bounds(
            price_min, price_max, parsed.price_min, parsed.price_max
        )
        eligible = apply_filters(
            self.products,
            categories=hard_categories or None,
            price_min=effective_min,
            price_max=effective_max,
        )
        if eligible.empty:
            return eligible.assign(text_score=pd.Series(dtype=float), final_score=pd.Series(dtype=float))
        query_vector = self.text_encoder.encode_query(query)
        all_scores = cosine_scores(self.text_matrix, query_vector)
        indices = eligible.index.to_numpy(dtype=int)
        candidate_scores = all_scores[indices] + self._soft_category_bonus(
            indices, soft_categories
        )
        order = np.argsort(candidate_scores)[::-1][:top_k]
        ranked_positions = indices[order]
        results = self.products.loc[ranked_positions].copy()
        results["text_score"] = all_scores[ranked_positions]
        results["image_score"] = np.nan
        results["final_score"] = candidate_scores[order]
        results["reason"] = results.apply(
            lambda row: (
                f"Phù hợp ngữ nghĩa truy vấn tiếng Việt; thuộc {row['category']}; "
                f"giá {int(row['price_vnd']):,} VND."
            ).replace(",", "."),
            axis=1,
        )
        results["parser_mode"] = parsed.parser_mode
        results["parser_detail"] = parsed.parser_detail
        results["parsed_query"] = json.dumps(parsed.to_dict(), ensure_ascii=False)
        status = self.llm.status()
        if status.ready:
            evidence = [
                {
                    "product_id": str(row.product_id),
                    "name": str(row.product_name),
                    "category": str(row.category),
                    "price_vnd": int(row.price_vnd),
                    "rating": float(row.rating),
                    "description": str(row.description).strip() if pd.notna(row.description) else "",
                    "review": top_review_evidence(row.reviews_text),
                    "text_score": float(row.text_score),
                    "recommendation_score": float(row.final_score),
                }
                for row in results.itertuples(index=False)
            ]
            try:
                reasons = self.llm.generate_reasons(query, evidence)
                results["reason"] = [
                    reasons.get(str(product_id), fallback)
                    for product_id, fallback in zip(results["product_id"], results["reason"], strict=True)
                ]
            except Exception:
                pass
        results = results.reset_index(drop=True)
        results.attrs["search_metadata"] = {
            "mode": "TEXT",
            "parser": parsed.to_dict(),
            "category_policy": category_policy,
            "effective_filters": {
                "categories": list(hard_categories),
                "soft_categories": list(soft_categories),
                "price_min": effective_min,
                "price_max": effective_max,
            },
            "candidate_count": len(eligible),
        }
        return results

    def search_image(
        self,
        image,
        *,
        category: str | None = None,
        price_min: int | None = None,
        price_max: int | None = None,
        top_k: int = 20,
    ) -> pd.DataFrame:
        query_vector = self.image_encoder.encode_image(image)
        explicit_category = self._selected_category(category)
        candidate_categories, category_metadata = self._image_category_candidates(
            query_vector,
            explicit_category=explicit_category,
        )
        eligible = apply_filters(
            self.products,
            categories=candidate_categories or None,
            price_min=price_min,
            price_max=price_max,
        )
        metadata = {
            "mode": "IMAGE",
            "category_inference": category_metadata,
            "effective_filters": {
                "categories": list(candidate_categories),
                "price_min": price_min,
                "price_max": price_max,
            },
            "candidate_count": len(eligible),
        }
        if eligible.empty:
            results = eligible.assign(
                text_score=pd.Series(dtype=float),
                image_score=pd.Series(dtype=float),
                final_score=pd.Series(dtype=float),
            )
            results.attrs["search_metadata"] = metadata
            return results
        all_scores = cosine_scores(self.image_matrix, query_vector)
        indices = eligible.index.to_numpy(dtype=int)
        ranked_positions = indices[np.argsort(all_scores[indices])[::-1]][:top_k]
        results = self.products.loc[ranked_positions].copy()
        results["text_score"] = np.nan
        results["image_score"] = all_scores[ranked_positions]
        results["final_score"] = results["image_score"]
        if category_metadata["uncertain"]:
            results["reason"] = results["category"].map(
                lambda category_name: (
                    f"Ảnh tương đồng với sản phẩm thuộc {category_name}; "
                    "nhận diện danh mục còn chưa chắc chắn."
                )
            )
        else:
            results["reason"] = results["category"].map(
                lambda category_name: (
                    f"Ảnh có đặc trưng thị giác gần với nhóm {category_name}."
                )
            )
        results = results.reset_index(drop=True)
        results.attrs["search_metadata"] = metadata
        return results

    def search_multimodal(
        self,
        query: str,
        image,
        *,
        category: str | None = None,
        price_min: int | None = None,
        price_max: int | None = None,
        top_k: int = 20,
        alpha: float | None = None,
    ) -> tuple[pd.DataFrame, dict]:
        parsed = self.query_parser.parse(query)
        effective_min, effective_max = self._effective_price_bounds(
            price_min, price_max, parsed.price_min, parsed.price_max
        )
        image_vector = self.image_encoder.encode_image(image)
        hard_categories, soft_categories, text_policy = self._text_category_policy(
            parsed,
            self._selected_category(category),
        )
        if hard_categories:
            candidate_categories = hard_categories
            category_metadata = {
                **text_policy,
                "primary_category": hard_categories[0],
                "candidate_categories": list(hard_categories),
                "uncertain": False,
            }
        elif soft_categories:
            candidate_categories = ()
            category_metadata = {
                **text_policy,
                "primary_category": None,
                "candidate_categories": list(soft_categories),
                "uncertain": True,
                "soft_bonus": CATEGORY_SOFT_BONUS,
            }
        else:
            candidate_categories, category_metadata = self._image_category_candidates(
                image_vector,
            )
        eligible = apply_filters(
            self.products,
            categories=candidate_categories or None,
            price_min=effective_min,
            price_max=effective_max,
        )
        metadata = {
            "mode": "MULTIMODAL",
            "parser": parsed.to_dict(),
            "alpha": SETTINGS.fusion_alpha if alpha is None else alpha,
            "llm_status": asdict(self.llm.status()),
            "category_inference": category_metadata,
            "effective_filters": {
                "categories": list(candidate_categories),
                "soft_categories": list(soft_categories),
                "price_min": effective_min,
                "price_max": effective_max,
            },
            "candidate_count": len(eligible),
        }
        if eligible.empty:
            return eligible, metadata
        text_vector = self.text_encoder.encode_query(query)
        all_text = cosine_scores(self.text_matrix, text_vector)
        all_image = cosine_scores(self.image_matrix, image_vector)
        indices = eligible.index.to_numpy(dtype=int)
        fused = fuse_scores(
            all_text[indices],
            all_image[indices],
            alpha=metadata["alpha"],
        )
        fused = fused + self._soft_category_bonus(indices, soft_categories)
        order = np.argsort(fused)[::-1][:top_k]
        ranked_positions = indices[order]
        results = self.products.loc[ranked_positions].copy()
        results["text_score"] = all_text[ranked_positions]
        results["image_score"] = all_image[ranked_positions]
        results["final_score"] = fused[order]
        results["reason"] = results.apply(
            lambda row: (
                f"Khớp cả ngữ nghĩa và hình ảnh; thuộc {row['category']}; "
                f"review thật: {top_review_evidence(row['reviews_text']) or 'chưa có trích đoạn ngắn'}."
            ),
            axis=1,
        )
        if metadata["llm_status"]["ready"]:
            evidence = [
                {
                    "product_id": str(row.product_id),
                    "name": str(row.product_name),
                    "category": str(row.category),
                    "price_vnd": int(row.price_vnd),
                    "rating": float(row.rating),
                    "description": str(row.description).strip() if pd.notna(row.description) else "",
                    "review": top_review_evidence(row.reviews_text),
                    "text_score": float(row.text_score),
                    "image_score": float(row.image_score),
                    "recommendation_score": float(row.final_score),
                }
                for row in results.itertuples(index=False)
            ]
            try:
                reasons = self.llm.generate_reasons(query, evidence)
                results["reason"] = [
                    reasons.get(str(product_id), fallback)
                    for product_id, fallback in zip(results["product_id"], results["reason"], strict=True)
                ]
            except Exception as exc:
                metadata["llm_reason_error"] = type(exc).__name__
        return results.reset_index(drop=True), metadata



