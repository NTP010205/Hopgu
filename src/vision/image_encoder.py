from __future__ import annotations

import io
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from src.config import MODELS_DIR, SETTINGS


MAX_UPLOAD_BYTES = 5 * 1024 * 1024
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class ImageValidationError(ValueError):
    pass


def validate_uploaded_image(
    content: bytes,
    *,
    filename: str | None = None,
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> Image.Image:
    if not content:
        raise ImageValidationError("File ảnh rỗng")
    if len(content) > max_bytes:
        raise ImageValidationError(
            f"Ảnh vượt giới hạn {max_bytes / (1024 * 1024):.0f} MB"
        )
    if filename:
        extension = Path(filename).suffix.lower()
        if extension not in ALLOWED_EXTENSIONS:
            raise ImageValidationError("Chỉ chấp nhận JPG, PNG hoặc WebP")
    try:
        with Image.open(io.BytesIO(content)) as probe:
            image_format = (probe.format or "").upper()
            if image_format not in ALLOWED_FORMATS:
                raise ImageValidationError(
                    f"Định dạng ảnh {image_format or 'không xác định'} không được hỗ trợ"
                )
            probe.verify()
        with Image.open(io.BytesIO(content)) as source:
            image = source.convert("RGB")
            image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
            return image.copy()
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ImageValidationError(f"Pillow không đọc được ảnh: {exc}") from exc


class ImageEncoder:
    """CLIP vision encoder; catalog và ảnh query dùng cùng không gian vector."""

    def __init__(self, model_name: str | None = None, *, offline: bool | None = None) -> None:
        self.model_name = model_name or SETTINGS.vision_model
        self.offline = SETTINGS.offline if offline is None else offline
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise RuntimeError("Thiếu transformers/torch; hãy cài requirements.txt") from exc
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self._processor = CLIPProcessor.from_pretrained(
            self.model_name,
            cache_dir=str(MODELS_DIR),
            local_files_only=self.offline,
        )
        self._model = CLIPModel.from_pretrained(
            self.model_name,
            cache_dir=str(MODELS_DIR),
            local_files_only=self.offline,
        )
        self._model.eval()
        self._torch = torch

    @property
    def model(self):
        self._load()
        return self._model

    @property
    def processor(self):
        self._load()
        return self._processor

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        if not images:
            raise ValueError("Danh sách ảnh không được rỗng")
        rgb_images = [image.convert("RGB") for image in images]
        inputs = self.processor(images=rgb_images, return_tensors="pt")
        with self._torch.inference_mode():
            # transformers 5.x trả BaseModelOutputWithPooling từ
            # get_image_features; chiếu pooler_output qua visual_projection
            # để lấy đúng không gian CLIP như các bản 4.x.
            vision_output = self.model.vision_model(pixel_values=inputs["pixel_values"])
            features = self.model.visual_projection(vision_output.pooler_output)
            features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return features.detach().cpu().numpy().astype(np.float32, copy=False)

    def encode_paths(self, paths: Sequence[Path]) -> np.ndarray:
        images: list[Image.Image] = []
        try:
            for path in paths:
                with Image.open(path) as source:
                    images.append(source.convert("RGB"))
            return self.encode_images(images)
        finally:
            for image in images:
                image.close()

    def encode_image(self, image: Image.Image) -> np.ndarray:
        return self.encode_images([image])[0]

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Encode text prompts into the same normalized CLIP space as images."""
        cleaned = [str(text).strip() for text in texts]
        if not cleaned or any(not text for text in cleaned):
            raise ValueError("Danh sách CLIP text prompt không được rỗng")
        inputs = self.processor(
            text=cleaned,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with self._torch.inference_mode():
            text_output = self.model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            )
            features = self.model.text_projection(text_output.pooler_output)
            features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return features.detach().cpu().numpy().astype(np.float32, copy=False)

