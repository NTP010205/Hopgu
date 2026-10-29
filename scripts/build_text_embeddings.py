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

from src.config import EMBEDDINGS_DIR, SETTINGS, ensure_data_directories
from src.data.loader import load_products
from src.nlp.text_encoder import TextEncoder, build_product_text


def text_fingerprint(product_ids: list[str], passages: list[str]) -> str:
    digest = hashlib.sha256()
    for product_id, passage in zip(product_ids, passages, strict=True):
        digest.update(product_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(passage.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def build(model_name: str, batch_size: int) -> dict:
    ensure_data_directories()
    products = load_products()
    product_ids = products["product_id"].astype(str).tolist()
    passages = [build_product_text(row) for _, row in products.iterrows()]
    encoder = TextEncoder(model_name=model_name)
    matrix = encoder.encode_passages(passages, batch_size=batch_size)
    if matrix.shape[0] != len(products):
        raise RuntimeError("Encoder trả sai số dòng embedding")
    np.save(EMBEDDINGS_DIR / "text_embeddings.npy", matrix, allow_pickle=False)
    manifest = {
        "kind": "text_embeddings",
        "model": model_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "product_ids": product_ids,
        "dataset_text_sha256": text_fingerprint(product_ids, passages),
        "shape": list(matrix.shape),
        "dtype": str(matrix.dtype),
        "normalized": True,
        "prefixes": {"query": "query: ", "passage": "passage: "},
    }
    (EMBEDDINGS_DIR / "text_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Precompute multilingual text embeddings")
    parser.add_argument("--model", default=SETTINGS.text_model)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    manifest = build(args.model, args.batch_size)
    print(json.dumps({key: value for key, value in manifest.items() if key != "product_ids"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

