from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EMBEDDINGS_DIR, PROJECT_ROOT, SETTINGS, ensure_data_directories
from src.data.loader import load_products
from src.vision.image_encoder import ImageEncoder


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(model_name: str, batch_size: int) -> dict:
    ensure_data_directories()
    products = load_products()
    product_ids = products["product_id"].astype(str).tolist()
    image_paths = [PROJECT_ROOT / str(path) for path in products["image_path"]]
    image_sha256 = [file_sha256(path) for path in image_paths]
    encoder = ImageEncoder(model_name=model_name)
    batches: list[np.ndarray] = []
    total_batches = (len(image_paths) + batch_size - 1) // batch_size
    for batch_index, start in enumerate(range(0, len(image_paths), batch_size), start=1):
        batch_paths = image_paths[start : start + batch_size]
        batches.append(encoder.encode_paths(batch_paths))
        print(f"IMAGE_BATCH {batch_index}/{total_batches} ({min(start + batch_size, len(image_paths))}/{len(image_paths)})", flush=True)
    matrix = np.concatenate(batches, axis=0)
    if matrix.shape[0] != len(products):
        raise RuntimeError("Encoder trả sai số dòng image embedding")
    np.save(EMBEDDINGS_DIR / "image_embeddings.npy", matrix, allow_pickle=False)
    digest = hashlib.sha256()
    for product_id, image_hash in zip(product_ids, image_sha256, strict=True):
        digest.update(product_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(image_hash.encode("ascii"))
        digest.update(b"\0")
    manifest = {
        "kind": "image_embeddings",
        "model": model_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "product_ids": product_ids,
        "image_sha256": image_sha256,
        "dataset_image_sha256": digest.hexdigest(),
        "shape": list(matrix.shape),
        "dtype": str(matrix.dtype),
        "normalized": True,
        "portable_paths": True,
    }
    (EMBEDDINGS_DIR / "image_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Precompute CLIP image embeddings")
    parser.add_argument("--model", default=SETTINGS.vision_model)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    manifest = build(args.model, args.batch_size)
    print(json.dumps({key: value for key, value in manifest.items() if key not in {"product_ids", "image_sha256"}}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

