from __future__ import annotations

import numpy as np


def cosine_scores(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    query = np.asarray(query, dtype=np.float32)
    if matrix.ndim != 2 or query.ndim != 1:
        raise ValueError("matrix phải 2D và query phải 1D")
    if matrix.shape[1] != query.shape[0]:
        raise ValueError("Embedding query và catalog khác số chiều")
    matrix_norm = np.linalg.norm(matrix, axis=1)
    query_norm = float(np.linalg.norm(query))
    if query_norm == 0 or np.any(matrix_norm == 0):
        raise ValueError("Embedding vector không được có norm bằng 0")
    return (matrix @ query) / (matrix_norm * query_norm)


def minmax_normalize(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32)
    if values.size == 0:
        return values.copy()
    low = float(np.min(values))
    high = float(np.max(values))
    if np.isclose(low, high):
        return np.ones_like(values) if high > 0 else np.zeros_like(values)
    return (values - low) / (high - low)

