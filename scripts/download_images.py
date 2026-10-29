from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import pandas as pd
import requests
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import IMAGES_DIR, PROCESSED_DIR, ensure_data_directories
from src.data.validator import validate_products, validate_reviews


MAX_SOURCE_BYTES = 12 * 1024 * 1024


def image_bytes_to_jpeg(content: bytes) -> bytes:
    if not content or len(content) > MAX_SOURCE_BYTES:
        raise ValueError("Ảnh nguồn rỗng hoặc vượt 12 MB")
    try:
        with Image.open(io.BytesIO(content)) as image:
            image.verify()
        with Image.open(io.BytesIO(content)) as image:
            converted = image.convert("RGB")
            converted.thumbnail((900, 900), Image.Resampling.LANCZOS)
            output = io.BytesIO()
            converted.save(output, format="JPEG", quality=90, optimize=True)
            return output.getvalue()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Pillow không đọc được ảnh: {exc}") from exc


def download_one(url: str, target: Path, *, timeout: float = 20.0) -> tuple[int, str]:
    response = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "Mozilla/5.0 (compatible; HopGuV2-Academic/1.0)"},
    )
    response.raise_for_status()
    content = response.content
    jpeg = image_bytes_to_jpeg(content)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(jpeg)
    return len(jpeg), hashlib.sha256(jpeg).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize(target_count: int, *, overwrite: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    product_path = PROCESSED_DIR / "products_staging.parquet"
    review_path = PROCESSED_DIR / "reviews_staging.parquet"
    if not product_path.is_file() or not review_path.is_file():
        raise FileNotFoundError("Thiếu staging parquet; chạy scripts/prepare_dataset.py trước")
    products = pd.read_parquet(product_path)
    reviews = pd.read_parquet(review_path)
    ensure_data_directories()

    successes: list[str] = []
    image_manifest: dict[str, dict[str, object]] = {}
    failures: dict[str, str] = {}
    for row in products.itertuples(index=False):
        if len(successes) >= target_count:
            break
        product_id = str(row.product_id)
        target = IMAGES_DIR / f"{product_id}.jpg"
        try:
            if target.is_file() and not overwrite:
                payload = target.read_bytes()
                image_bytes_to_jpeg(payload)
                size = len(payload)
                digest = hashlib.sha256(payload).hexdigest()
            else:
                size, digest = download_one(str(row.image_url), target)
            successes.append(product_id)
            image_manifest[product_id] = {
                "path": f"data/images/{product_id}.jpg",
                "sha256": digest,
                "bytes": size,
                "source_url": str(row.image_url),
            }
            print(f"OK {len(successes):04d}/{target_count}: {product_id}")
        except Exception as exc:  # báo từng URL hỏng nhưng tiếp tục dùng candidate kế tiếp
            failures[product_id] = f"{type(exc).__name__}: {exc}"
            print(f"FAIL {product_id}: {failures[product_id]}")

    if len(successes) < target_count:
        raise RuntimeError(
            f"Chỉ tải được {len(successes)}/{target_count} ảnh hợp lệ; cần tạo staging lớn hơn"
        )

    final_products = products.loc[products["product_id"].astype(str).isin(successes)].copy()
    order = {product_id: index for index, product_id in enumerate(successes)}
    final_products["_order"] = final_products["product_id"].astype(str).map(order)
    final_products = final_products.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
    final_reviews = reviews.loc[
        reviews["product_id"].astype(str).isin(set(successes))
    ].reset_index(drop=True)
    validate_products(final_products).raise_for_errors()
    validate_reviews(final_reviews, set(final_products["product_id"].astype(str))).raise_for_errors()

    final_products.to_parquet(PROCESSED_DIR / "products.parquet", index=False)
    final_reviews.to_parquet(PROCESSED_DIR / "reviews.parquet", index=False)
    manifest = {
        "kind": "local_image_snapshot",
        "products": len(final_products),
        "reviews": len(final_reviews),
        "images": image_manifest,
        "failures": failures,
        "staging_manifest_sha256": file_sha256(
            PROCESSED_DIR / "dataset_staging_manifest.json"
        ),
        "products_parquet_sha256": file_sha256(
            PROCESSED_DIR / "products.parquet"
        ),
        "reviews_parquet_sha256": file_sha256(
            PROCESSED_DIR / "reviews.parquet"
        ),
    }
    (PROCESSED_DIR / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return final_products, final_reviews, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tạo snapshot ảnh local từ URL trong staging")
    parser.add_argument("--target-count", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    products, reviews, manifest = materialize(args.target_count, overwrite=args.overwrite)
    print(
        f"Hoàn tất: {len(products)} sản phẩm, {len(reviews)} review, "
        f"{len(manifest['failures'])} ảnh lỗi bị loại"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

