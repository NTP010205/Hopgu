from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.config import MODELS_DIR, SETTINGS


def build_product_text(product: object, *, max_review_chars: int = 1800) -> str:
    """Tạo passage chỉ từ evidence thật của một sản phẩm."""
    get = product.get if hasattr(product, "get") else lambda key, default="": getattr(product, key, default)
    reviews = get("reviews_text", [])
    if not isinstance(reviews, (list, tuple, np.ndarray)):
        reviews = [str(reviews)] if reviews else []
    review_text = " ".join(str(review).strip() for review in reviews if str(review).strip())
    review_text = review_text[:max_review_chars]
    parts = [
        f"Tên sản phẩm: {str(get('product_name', '')).strip()}",
        f"Danh mục: {str(get('category', '')).strip()}",
        f"Mô tả: {str(get('description', '')).strip()}",
        f"Review khách hàng: {review_text}",
    ]
    return " | ".join(part for part in parts if part.split(":", 1)[-1].strip())


class TextEncoder:
    """SentenceTransformer đa ngôn ngữ với quy ước query/passage của E5."""

    def __init__(self, model_name: str | None = None, *, offline: bool | None = None) -> None:
        self.model_name = model_name or SETTINGS.text_model
        self.offline = SETTINGS.offline if offline is None else offline
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "Thiếu sentence-transformers. Hãy cài requirements.txt trước."
                ) from exc
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            self._model = SentenceTransformer(
                self.model_name,
                cache_folder=str(MODELS_DIR),
                local_files_only=self.offline,
            )
        return self._model

    def encode_queries(self, queries: Sequence[str], *, batch_size: int = 32) -> np.ndarray:
        texts = [f"query: {str(query).strip()}" for query in queries]
        if not all(text.removeprefix("query: ").strip() for text in texts):
            raise ValueError("Query không được rỗng")
        return np.asarray(
            self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )

    def encode_passages(self, passages: Sequence[str], *, batch_size: int = 32) -> np.ndarray:
        texts = [f"passage: {str(passage).strip()}" for passage in passages]
        if not all(text.removeprefix("passage: ").strip() for text in texts):
            raise ValueError("Passage không được rỗng")
        return np.asarray(
            self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=True,
            ),
            dtype=np.float32,
        )

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode_queries([query], batch_size=1)[0]

