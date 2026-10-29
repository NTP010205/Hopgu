from __future__ import annotations

import numpy as np

from src.recommender.scoring import minmax_normalize


def fuse_scores(text_scores: np.ndarray, image_scores: np.ndarray, *, alpha: float = 0.6) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha phải nằm trong [0, 1]")
    text_values = np.asarray(text_scores, dtype=np.float32)
    image_values = np.asarray(image_scores, dtype=np.float32)
    if text_values.shape != image_values.shape:
        raise ValueError("text_score và image_score phải cùng shape")
    return alpha * minmax_normalize(text_values) + (1.0 - alpha) * minmax_normalize(image_values)

