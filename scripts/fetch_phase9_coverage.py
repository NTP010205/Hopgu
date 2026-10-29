from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.download_images import download_one
from src.config import EVAL_DIR, PROCESSED_DIR, RAW_DIR


API_ROOT = "https://api.tiki.vn/v2"
USER_AGENT = "Mozilla/5.0 (compatible; HopGuV2-Academic/1.0)"
PRODUCT_COLUMNS = pd.read_csv(
    RAW_DIR / "tiki_products_clean_full.csv", nrows=0
).columns.tolist()
DETAIL_COLUMNS = pd.read_csv(
    RAW_DIR / "tiki_product_details.csv", nrows=0
).columns.tolist()
REVIEW_COLUMNS = pd.read_csv(
    RAW_DIR / "tiki_reviews.csv", nrows=0
).columns.tolist()

CATALOG_GROUPS = (
    {
        "group": "handbag",
        "queries": ("túi xách nữ", "túi đeo chéo nữ"),
        "target": 20,
        "broad_category": "Thời trang nữ",
        "pattern": r"\btúi\s*(?:xách|đeo\s*chéo|đeo\s*vai)\b|\bhandbag\b|\bcrossbody\b",
    },
    {
        "group": "backpack",
        "queries": ("balo laptop", "balo đi học"),
        "target": 20,
        "broad_category": "Điện tử & Phụ kiện",
        "pattern": r"\bbalo\b|\bba\s*lô\b|\bbackpack\b",
    },
    {
        "group": "shoes_men",
        "queries": ("giày sneaker nam",),
        "target": 10,
        "broad_category": "Thời trang nam",
        "pattern": r"\bgiày\b|\bsneaker\b",
        "exclude_pattern": r"\bgiấy\s*lau\b|\bvệ\s*sinh\b|\bphụ\s*kiện\b|\btủ\s*giày\b|\bkệ\s*giày\b",
    },
    {
        "group": "shoes_women",
        "queries": ("giày sneaker nữ", "giày thể thao nữ", "giày chạy bộ nữ"),
        "target": 10,
        "broad_category": "Thời trang nữ",
        "pattern": r"\bgiày\b|\bsneaker\b",
        "exclude_pattern": r"\bgiấy\s*lau\b|\bvệ\s*sinh\b|\bphụ\s*kiện\b|\btủ\s*giày\b|\bkệ\s*giày\b|\bbé\s*gái\b|\btrẻ\s*em\b",
    },
)

HOLDOUT_GROUPS = (
    ("bag", "túi xách nữ", "Thời trang nữ", r"\btúi\s*(?:xách|đeo\s*chéo|đeo\s*vai)\b"),
    ("shoes", "giày sneaker nam", "Thời trang nam", r"\bgiày\b|\bsneaker\b"),
    ("headphones", "tai nghe bluetooth", "Điện tử & Phụ kiện", r"\btai\s*nghe\b|\bheadphones?\b"),
    ("cosmetics", "kem chống nắng", "Làm đẹp & Sức khỏe", r"\bkem\s*chống\s*nắng\b|\bmỹ\s*phẩm\b"),
    ("home", "bình giữ nhiệt", "Nhà cửa & Đời sống", r"\bbình\s*giữ\s*nhiệt\b"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False) if value is not None else ""


def scalar_quantity(value: Any) -> int:
    if isinstance(value, dict):
        value = value.get("value", 0)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


class TikiClient:
    def __init__(self, *, delay: float = 0.08) -> None:
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})

    def get_json(self, path: str, *, params: dict | None = None) -> dict:
        last_error: Exception | None = None
        for attempt in range(6):
            try:
                response = self.session.get(
                    f"{API_ROOT}/{path.lstrip('/')}",
                    params=params,
                    timeout=30,
                )
                response.raise_for_status()
                payload = response.json()
                time.sleep(self.delay)
                return payload
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Tiki API thất bại sau retry: {last_error}")

    def search(self, query: str, *, pages: int = 4) -> list[dict]:
        rows: list[dict] = []
        for page in range(1, pages + 1):
            try:
                payload = self.get_json(
                    "products",
                    params={"limit": 40, "page": page, "q": query},
                )
            except RuntimeError:
                if rows:
                    break
                raise
            current = payload.get("data", [])
            if not isinstance(current, list) or not current:
                break
            rows.extend(item for item in current if isinstance(item, dict))
        return rows

    def detail(self, product_id: str) -> dict:
        return self.get_json(f"products/{product_id}")

    def reviews(self, product_id: str) -> list[dict]:
        payload = self.get_json(
            "reviews",
            params={"limit": 5, "include": "comments", "product_id": product_id},
        )
        rows = payload.get("data", [])
        return rows if isinstance(rows, list) else []


def product_row(detail: dict, search_row: dict, broad_category: str, now: str) -> dict:
    category = detail.get("categories") or {}
    if not isinstance(category, dict):
        category = {}
    product_url = detail.get("url_path") or search_row.get("url_path") or ""
    if product_url and not str(product_url).startswith("http"):
        product_url = f"https://tiki.vn/{str(product_url).lstrip('/')}"
    row = {
        "product_id": str(detail.get("id") or search_row["id"]),
        "name": detail.get("name") or search_row.get("name"),
        "sku": detail.get("sku") or search_row.get("sku"),
        "category_id": category.get("id"),
        "category_name": category.get("name") or search_row.get("category_name") or broad_category,
        "brand_id": (detail.get("brand") or {}).get("id") if isinstance(detail.get("brand"), dict) else search_row.get("brand_id"),
        "brand_name": (detail.get("brand") or {}).get("name") if isinstance(detail.get("brand"), dict) else search_row.get("brand_name"),
        "price": detail.get("price") or search_row.get("price"),
        "original_price": detail.get("original_price") or search_row.get("original_price"),
        "discount": detail.get("discount") or search_row.get("discount"),
        "discount_rate": detail.get("discount_rate") or search_row.get("discount_rate"),
        "quantity_sold": scalar_quantity(detail.get("quantity_sold") or search_row.get("quantity_sold")),
        "rating_average": detail.get("rating_average") or search_row.get("rating_average") or 0,
        "review_count": detail.get("review_count") or search_row.get("review_count") or 0,
        "seller_id": search_row.get("seller_id"),
        "seller_name": search_row.get("seller_name"),
        "inventory_status": detail.get("inventory_status") or search_row.get("inventory_status"),
        "short_description": detail.get("short_description") or search_row.get("short_description") or "",
        "product_url": product_url,
        "thumbnail_url": detail.get("thumbnail_url") or search_row.get("thumbnail_url"),
        "data_source": "tiki_api_phase9",
        "crawled_at": now,
        "phase9_broad_category": broad_category,
    }
    return {column: row.get(column) for column in [*PRODUCT_COLUMNS, "phase9_broad_category"]}


def detail_row(detail: dict, product_id: str, now: str) -> dict:
    images = detail.get("images") or []
    image_urls = [
        item.get("large_url") or item.get("base_url") or item.get("medium_url")
        for item in images
        if isinstance(item, dict)
    ]
    row = {
        "product_id": product_id,
        "short_description": detail.get("short_description") or "",
        "description_html": detail.get("description") or "",
        "specifications": json_text(detail.get("specifications")),
        "images": json_text([url for url in image_urls if url]),
        "warranty_info": detail.get("warranty_info") or "",
        "return_policy": detail.get("return_policy") or "",
        "all_categories": json_text(detail.get("all_categories")),
        "breadcrumbs": json_text(detail.get("breadcrumbs")),
        "configurable_options": json_text(detail.get("configurable_options")),
        "crawled_at": now,
        "data_source": "tiki_api_phase9",
    }
    return {column: row.get(column) for column in DETAIL_COLUMNS}


def review_rows(items: list[dict], product_id: str, now: str) -> list[dict]:
    rows: list[dict] = []
    for item in items:
        content = re.sub(r"\s+", " ", str(item.get("content") or "")).strip()
        if not content:
            continue
        creator = item.get("created_by") or {}
        seller = item.get("seller") or {}
        created_at = item.get("created_at")
        if isinstance(created_at, (int, float)):
            created_at = datetime.fromtimestamp(created_at, tz=timezone.utc).isoformat()
        images = [
            image.get("full_path")
            for image in (item.get("images") or [])
            if isinstance(image, dict) and image.get("full_path")
        ]
        row = {
            "review_id": str(item.get("id")),
            "product_id": product_id,
            "customer_id": creator.get("id"),
            "customer_name": creator.get("name") or creator.get("full_name"),
            "rating": item.get("rating") or 0,
            "title": item.get("title") or "",
            "content": content,
            "thank_count": item.get("thank_count") or 0,
            "is_purchased": creator.get("purchased"),
            "has_images": bool(images),
            "images": json_text(images),
            "seller_id": seller.get("id"),
            "seller_name": seller.get("name"),
            "product_attributes": json_text(item.get("product_attributes")),
            "vote_agree": json_text((item.get("vote_attributes") or {}).get("agree")),
            "vote_disagree": json_text((item.get("vote_attributes") or {}).get("disagree")),
            "usage_duration": (item.get("timeline") or {}).get("content"),
            "sentiment": "",
            "created_at": created_at,
            "crawled_at": now,
        }
        rows.append({column: row.get(column) for column in REVIEW_COLUMNS})
    return rows


def eligible_search_row(
    row: dict,
    pattern: str,
    excluded_ids: set[str],
    exclude_pattern: str | None = None,
) -> bool:
    product_id = str(row.get("id") or "")
    text = f"{row.get('name') or ''} {row.get('category_name') or ''}"
    return bool(
        product_id
        and product_id not in excluded_ids
        and re.search(pattern, text, flags=re.IGNORECASE)
        and not (
            exclude_pattern
            and re.search(exclude_pattern, text, flags=re.IGNORECASE)
        )
        and int(row.get("review_count") or 0) > 0
        and float(row.get("price") or 0) > 0
        and str(row.get("thumbnail_url") or "").startswith("http")
    )


def collect_catalog(
    client: TikiClient,
    existing_ids: set[str],
    checkpoint_path: Path,
) -> tuple[list[dict], list[dict], list[dict], dict[str, list[str]]]:
    if checkpoint_path.is_file():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        product_rows = checkpoint["products"]
        detail_rows = checkpoint["details"]
        all_reviews = checkpoint["reviews"]
        selected_by_group = checkpoint["selected_by_group"]
        product_by_id = {str(row["product_id"]): row for row in product_rows}
        rejected_ids: set[str] = set()
        for group in CATALOG_GROUPS:
            exclude_pattern = group.get("exclude_pattern")
            if not exclude_pattern:
                continue
            kept: list[str] = []
            for product_id in selected_by_group.get(group["group"], []):
                row = product_by_id.get(str(product_id), {})
                text = f"{row.get('name') or ''} {row.get('category_name') or ''}"
                if re.search(exclude_pattern, text, flags=re.IGNORECASE):
                    rejected_ids.add(str(product_id))
                else:
                    kept.append(str(product_id))
            selected_by_group[group["group"]] = kept
        if rejected_ids:
            product_rows = [
                row for row in product_rows
                if str(row["product_id"]) not in rejected_ids
            ]
            detail_rows = [
                row for row in detail_rows
                if str(row["product_id"]) not in rejected_ids
            ]
            all_reviews = [
                row for row in all_reviews
                if str(row["product_id"]) not in rejected_ids
            ]
            print(f"CHECKPOINT REJECTED {sorted(rejected_ids)}", flush=True)
        print(f"RESUME {checkpoint_path}", flush=True)
    else:
        product_rows = []
        detail_rows = []
        all_reviews = []
        selected_by_group = {}
    selected_ids = set(existing_ids) | {
        str(row["product_id"]) for row in product_rows
    }
    now = datetime.now(timezone.utc).isoformat()
    for group in CATALOG_GROUPS:
        candidates: list[dict] = []
        seen: set[str] = set()
        for query in group["queries"]:
            try:
                search_rows = client.search(query)
            except RuntimeError as exc:
                print(
                    f"SEARCH RETRY LATER {group['group']} {query}: {exc}",
                    flush=True,
                )
                continue
            for row in search_rows:
                product_id = str(row.get("id") or "")
                if product_id not in seen:
                    seen.add(product_id)
                    candidates.append(row)
        selected_by_group.setdefault(group["group"], [])
        if len(selected_by_group[group["group"]]) >= group["target"]:
            continue
        for row in candidates:
            if len(selected_by_group[group["group"]]) >= group["target"]:
                break
            if not eligible_search_row(
                row,
                group["pattern"],
                selected_ids,
                group.get("exclude_pattern"),
            ):
                continue
            product_id = str(row["id"])
            try:
                detail = client.detail(product_id)
                reviews = review_rows(client.reviews(product_id), product_id, now)
            except Exception as exc:
                print(f"SKIP {group['group']} {product_id}: {type(exc).__name__}: {exc}", flush=True)
                continue
            if not reviews:
                continue
            mapped = product_row(detail, row, group["broad_category"], now)
            if not mapped.get("thumbnail_url") or float(mapped.get("price") or 0) <= 0:
                continue
            product_rows.append(mapped)
            detail_rows.append(detail_row(detail, product_id, now))
            all_reviews.extend(reviews)
            selected_ids.add(product_id)
            selected_by_group[group["group"]].append(product_id)
            checkpoint_path.write_text(
                json.dumps(
                    {
                        "products": product_rows,
                        "details": detail_rows,
                        "reviews": all_reviews,
                        "selected_by_group": selected_by_group,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            print(
                f"CATALOG {group['group']} "
                f"{len(selected_by_group[group['group']]):02d}/{group['target']}: "
                f"{product_id} {mapped['name']}",
                flush=True,
            )
        if len(selected_by_group[group["group"]]) < group["target"]:
            raise RuntimeError(
                f"Không đủ sản phẩm {group['group']} có review: "
                f"{len(selected_by_group[group['group']])}/{group['target']}"
            )
    return product_rows, detail_rows, all_reviews, selected_by_group


def collect_holdout(
    client: TikiClient,
    excluded_ids: set[str],
    *,
    per_group: int = 4,
) -> list[dict]:
    output_dir = EVAL_DIR / "out_of_catalog_images"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    used = set(excluded_ids)
    for group_name, query, expected_category, pattern in HOLDOUT_GROUPS:
        selected = 0
        for item in client.search(query, pages=5):
            product_id = str(item.get("id") or "")
            if product_id in used:
                continue
            text = f"{item.get('name') or ''} {item.get('category_name') or ''}"
            if not re.search(pattern, text, flags=re.IGNORECASE):
                continue
            if group_name == "shoes" and re.search(
                r"\bgiấy\s*lau\b|\bvệ\s*sinh\b|\bphụ\s*kiện\b|\btủ\s*giày\b|\bkệ\s*giày\b",
                text,
                flags=re.IGNORECASE,
            ):
                continue
            image_url = str(item.get("thumbnail_url") or "")
            if not image_url.startswith("http"):
                continue
            path = output_dir / f"{group_name}_{product_id}.jpg"
            try:
                size, digest = download_one(image_url, path)
            except Exception as exc:
                print(f"HOLDOUT SKIP {product_id}: {type(exc).__name__}: {exc}", flush=True)
                continue
            rows.append(
                {
                    "group": group_name,
                    "product_id": product_id,
                    "product_name": str(item.get("name") or ""),
                    "expected_category": expected_category,
                    "image_path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "image_url": image_url,
                    "image_sha256": digest,
                    "image_bytes": size,
                    "source": "Tiki public API search result; excluded from catalog",
                    "query_used_for_collection": query,
                }
            )
            used.add(product_id)
            selected += 1
            print(f"HOLDOUT {group_name} {selected:02d}/{per_group}: {product_id}", flush=True)
            if selected >= per_group:
                break
        if selected < per_group:
            raise RuntimeError(f"Không đủ holdout {group_name}: {selected}/{per_group}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Bổ sung coverage Tiki thật cho Phase 9")
    parser.add_argument("--delay", type=float, default=0.08)
    args = parser.parse_args()

    existing = pd.read_parquet(PROCESSED_DIR / "products.parquet")
    existing_ids = set(existing["product_id"].astype(str))
    client = TikiClient(delay=args.delay)
    checkpoint_path = RAW_DIR / ".phase9_coverage_checkpoint.json"
    products, details, reviews, selected_by_group = collect_catalog(
        client,
        existing_ids,
        checkpoint_path,
    )
    added_ids = {str(row["product_id"]) for row in products}
    holdout = collect_holdout(client, existing_ids | added_ids)

    outputs = {
        "products": RAW_DIR / "tiki_phase9_coverage_products.csv",
        "details": RAW_DIR / "tiki_phase9_coverage_details.csv",
        "reviews": RAW_DIR / "tiki_phase9_coverage_reviews.csv",
    }
    pd.DataFrame(products).to_csv(outputs["products"], index=False, encoding="utf-8-sig")
    pd.DataFrame(details).to_csv(outputs["details"], index=False, encoding="utf-8-sig")
    pd.DataFrame(reviews).to_csv(outputs["reviews"], index=False, encoding="utf-8-sig")

    seed_categories = dict(
        zip(existing["product_id"].astype(str), existing["category"].astype(str), strict=True)
    )
    manifest = {
        "kind": "phase9_tiki_live_coverage",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "api_root": API_ROOT,
        "user_agent": USER_AGENT,
        "catalog_before_count": len(existing),
        "preserved_phase8_ids": existing["product_id"].astype(str).tolist(),
        "preserved_phase8_categories": seed_categories,
        "coverage_product_ids": [str(row["product_id"]) for row in products],
        "coverage_by_group": selected_by_group,
        "holdout_product_ids": [row["product_id"] for row in holdout],
        "holdout_note": "Không thuộc catalog; chỉ dùng test sau cùng, không train/tune.",
        "rows": {
            "products": len(products),
            "details": len(details),
            "reviews": len(reviews),
            "holdout_images": len(holdout),
        },
        "raw_sha256": {key: sha256(path) for key, path in outputs.items()},
    }
    manifest_path = RAW_DIR / "phase9_coverage_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    holdout_path = EVAL_DIR / "out_of_catalog_manifest.json"
    holdout_path.write_text(
        json.dumps(
            {
                "generated_at": manifest["generated_at"],
                "source": "Tiki public API",
                "usage": "Phase 9 held-out evaluation only; not used for training or tuning.",
                "items": holdout,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if checkpoint_path.is_file():
        checkpoint_path.unlink()
    print(f"WROTE {manifest_path}")
    print(f"WROTE {holdout_path}")
    print(json.dumps(manifest["rows"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
