from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
IMAGES_DIR = DATA_DIR / "images"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
EVAL_DIR = DATA_DIR / "eval"
EVAL_RESULTS_DIR = EVAL_DIR / "results"
MODELS_DIR = DATA_DIR / "models"


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Biến môi trường {name} phải là số, nhận được: {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    text_model: str = os.getenv("HOPGU_TEXT_MODEL", "intfloat/multilingual-e5-small")
    vision_model: str = os.getenv("HOPGU_VISION_MODEL", "openai/clip-vit-base-patch32")
    ollama_url: str = os.getenv("HOPGU_OLLAMA_URL", "http://127.0.0.1:11434")
    ollama_model: str = os.getenv("HOPGU_OLLAMA_MODEL", "qwen2.5:3b-instruct")
    fusion_alpha: float = _env_float("HOPGU_FUSION_ALPHA", 0.8)
    offline: bool = os.getenv("HOPGU_OFFLINE", "0").strip().lower() in {"1", "true", "yes"}

    def __post_init__(self) -> None:
        if not 0.0 <= self.fusion_alpha <= 1.0:
            raise ValueError("HOPGU_FUSION_ALPHA phải nằm trong [0, 1]")


SETTINGS = Settings()


def ensure_data_directories() -> None:
    for path in (
        RAW_DIR,
        PROCESSED_DIR,
        IMAGES_DIR,
        EMBEDDINGS_DIR,
        EVAL_DIR,
        EVAL_RESULTS_DIR,
        MODELS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)

