from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DIR, RAW_DIR, ensure_data_directories
from src.data.validator import validate_products, validate_reviews


SOURCE_FILES = {
    "products": "tiki_products_clean_full.csv",
    "details": "tiki_product_details.csv",
    "reviews": "tiki_reviews.csv",
}
SUPPLEMENT_FILES = {
    "products": "tiki_phase9_coverage_products.csv",
    "details": "tiki_phase9_coverage_details.csv",
    "reviews": "tiki_phase9_coverage_reviews.csv",
}


def ascii_fold(value: object) -> str:
    text = unicodedata.normalize("NFD", str(value).lower())
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


CATEGORY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Sách", ("sach", "truyen", "tieu thuyet", "van hoc", "tap chi", "tu dien", "giao khoa")),
    ("Làm đẹp & Sức khỏe", ("kem ", "sua rua mat", "dau goi", "sua tam", "son ", "trang diem", "khu mui", "cham soc", "khau trang", "bao cao su", "bang ve sinh", "thuc pham chuc nang")),
    ("Thời trang nữ", ("ao nang nguc", "ao bra", "quan ao nu", "dam nu", "vay nu", "tui xach nu", "giay nu", "dep nu", "noi y nu")),
    ("Thời trang nam", ("ao thun nam", "ao so mi nam", "quan short", "quan lot", "quan nam", "giay nam", "dep nam", "thoi trang nam")),
    ("Điện tử & Phụ kiện", ("dien thoai", "may tinh bang", "tai nghe", "chuot ", "ban phim", "cu sac", "adapter sac", "day cap", "cap sac", "op lung", "bao da", "pin ", "loa ", "camera", "dong ho thong minh", "thiet bi so")),
    ("Điện gia dụng", ("noi ", "noi com", "noi chien", "may xay", "may say", "may pha", "may lam", "ban ui", "binh dun", "quat ", "hut bui", "dien gia dung", "may loc", "bep ", "may ep")),
    ("Thể thao & Dã ngoại", ("the thao", "chay bo", "gym", "bong ", "vot ", "da ngoai", "cam trai", "xe dap", "tui ngu")),
    ("Bách hóa", ("sua ", "do uong", "banh ", "keo ", "gao ", "ca phe", "tra ", "gia vi", "thuc pham", "bach hoa", "dau an", "dau dau nanh", "dau ")),
    ("Nhà cửa & Đời sống", ("binh giu nhiet", "giu nhiet", "hop com", "ly ", "coc ", "khan ", "ga ", "goi ", "nem ", "chao ", "nha cua", "doi song", "phong an", "nha bep", "den ", "hop dung")),
)


def broad_category(category_name: object, product_name: object) -> str:
    category_text = f" {ascii_fold(category_name)} "
    name_text = f" {ascii_fold(product_name)} "
    fashion_words = (" ao ", " quan ", " dam ", " vay ", " do lot ", " noi y ", " giay ", " dep ")
    if any(word in category_text or word in name_text for word in fashion_words):
        if " nu " in category_text or " nu " in name_text:
            return "Thời trang nữ"
        if " nam " in category_text or " nam " in name_text:
            return "Thời trang nam"
    for candidate_text in (category_text, name_text):
        for label, needles in CATEGORY_RULES:
            if any(needle in candidate_text for needle in needles):
                return label
    return "Khác"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _balanced_take(products: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Lấy round-robin theo danh mục để nhóm lớn không lấn át nhóm nhỏ."""
    groups = {
        category: group.index.tolist()
        for category, group in products.groupby("category", sort=True)
    }
    selected_indices: list[int] = []
    offset = 0
    while len(selected_indices) < limit:
        added = False
        for category in sorted(groups):
            indices = groups[category]
            if offset < len(indices):
                selected_indices.append(indices[offset])
                added = True
                if len(selected_indices) >= limit:
                    break
        if not added:
            break
        offset += 1
    return products.loc[selected_indices].copy()


def build_dataset(
    limit: int,
    per_category: int | None = None,
    min_category_size: int = 1,
    selection_manifest: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = {key: RAW_DIR / name for key, name in SOURCE_FILES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Thiếu dữ liệu raw: " + ", ".join(missing))

    products = pd.read_csv(paths["products"], dtype={"product_id": "string"})
    details = pd.read_csv(paths["details"], dtype={"product_id": "string"})
    reviews = pd.read_csv(paths["reviews"], dtype={"product_id": "string", "review_id": "string"})

    selection: dict | None = None
    if selection_manifest is not None:
        if not selection_manifest.is_file():
            raise FileNotFoundError(f"Thiếu selection manifest: {selection_manifest}")
        selection = json.loads(selection_manifest.read_text(encoding="utf-8"))
        supplement_paths = {
            key: RAW_DIR / name for key, name in SUPPLEMENT_FILES.items()
        }
        missing_supplement = [
            str(path) for path in supplement_paths.values() if not path.is_file()
        ]
        if missing_supplement:
            raise FileNotFoundError(
                "Thiếu dữ liệu coverage Phase 9: " + ", ".join(missing_supplement)
            )
        products = pd.concat(
            [
                products,
                pd.read_csv(
                    supplement_paths["products"], dtype={"product_id": "string"}
                ),
            ],
            ignore_index=True,
            sort=False,
        )
        details = pd.concat(
            [
                details,
                pd.read_csv(
                    supplement_paths["details"], dtype={"product_id": "string"}
                ),
            ],
            ignore_index=True,
            sort=False,
        )
        reviews = pd.concat(
            [
                reviews,
                pd.read_csv(
                    supplement_paths["reviews"],
                    dtype={"product_id": "string", "review_id": "string"},
                ),
            ],
            ignore_index=True,
            sort=False,
        )

    products["product_id"] = products["product_id"].astype(str)
    details["product_id"] = details["product_id"].astype(str)
    reviews["product_id"] = reviews["product_id"].astype(str)
    reviews["content"] = reviews["content"].map(_clean_text)
    reviews = reviews.loc[reviews["content"].ne("")].copy()
    reviews = reviews.drop_duplicates(subset=["review_id"], keep="first")

    detail_columns = details[["product_id", "description_html", "short_description"]].rename(
        columns={"short_description": "detail_short_description"}
    )
    products = products.merge(detail_columns, on="product_id", how="left")
    products = products.loc[
        products["product_id"].isin(reviews["product_id"])
        & products["name"].notna()
        & products["category_name"].notna()
        & products["thumbnail_url"].fillna("").str.startswith("http")
        & (pd.to_numeric(products["price"], errors="coerce") > 0)
    ].copy()
    products["category"] = [
        broad_category(category, name)
        for category, name in products[["category_name", "name"]].itertuples(index=False)
    ]
    if selection is not None:
        preserved_categories = {
            str(key): str(value)
            for key, value in selection["preserved_phase8_categories"].items()
        }
        products.loc[
            products["product_id"].isin(preserved_categories), "category"
        ] = products.loc[
            products["product_id"].isin(preserved_categories), "product_id"
        ].map(preserved_categories)
        if "phase9_broad_category" in products.columns:
            supplement_mask = products["phase9_broad_category"].notna()
            products.loc[supplement_mask, "category"] = products.loc[
                supplement_mask, "phase9_broad_category"
            ].astype(str)
    products = products.loc[products["category"].ne("Khác")].copy()

    review_rank = reviews.sort_values(
        by=["thank_count", "rating"], ascending=[False, False], na_position="last"
    )
    review_lists = review_rank.groupby("product_id")["content"].apply(lambda values: list(values.head(5)))

    products["review_count_real"] = products["product_id"].map(reviews.groupby("product_id").size())
    products["reviews_text"] = products["product_id"].map(review_lists)
    products["description"] = products["description_html"].map(_clean_text)
    fallback_description = products["detail_short_description"].map(_clean_text)
    products.loc[products["description"].eq(""), "description"] = fallback_description
    source_short = products["short_description"].map(_clean_text)
    products.loc[products["description"].eq(""), "description"] = source_short

    products = products.sort_values(
        by=["category", "review_count_real", "rating_average", "product_id"],
        ascending=[True, False, False, True],
    )
    # Snapshot có thể chứa nhiều seller_product cho cùng một product_id. Schema cuối
    # yêu cầu định danh sản phẩm duy nhất, vì vậy giữ record được xếp hạng cao nhất.
    products = products.drop_duplicates(subset=["product_id"], keep="first")
    category_sizes = products["category"].value_counts()
    eligible_categories = category_sizes[category_sizes >= min_category_size].index
    products = products.loc[products["category"].isin(eligible_categories)].copy()
    if selection is not None:
        desired_ids = [
            *map(str, selection["preserved_phase8_ids"]),
            *map(str, selection["coverage_product_ids"]),
        ]
        if len(desired_ids) != len(set(desired_ids)):
            raise ValueError("Selection manifest chứa product_id trùng")
        if limit != len(desired_ids):
            raise ValueError(
                f"--limit phải bằng {len(desired_ids)} khi dùng selection manifest"
            )
        indexed = products.set_index("product_id", drop=False)
        missing_selected = [value for value in desired_ids if value not in indexed.index]
        if missing_selected:
            raise ValueError(
                f"Selection manifest có {len(missing_selected)} ID không hợp lệ: "
                + ", ".join(missing_selected[:10])
            )
        selected = indexed.loc[desired_ids].copy()
    else:
        if per_category is not None:
            products = products.groupby("category", group_keys=False, sort=True).head(per_category)
        selected = _balanced_take(products, limit)

    selected["image_path"] = selected["product_id"].map(lambda value: f"data/images/{value}.jpg")
    selected["source"] = "Tiki"
    selected["price_vnd"] = pd.to_numeric(selected["price"], errors="raise").round().astype("int64")
    selected["rating"] = pd.to_numeric(selected["rating_average"], errors="coerce").fillna(0.0).astype(float)
    selected["review_count"] = selected["review_count_real"].astype("int64")
    selected = selected.rename(columns={"name": "product_name", "thumbnail_url": "image_url"})

    output_columns = [
        "product_id", "product_name", "category", "price_vnd", "description", "rating",
        "review_count", "reviews_text", "image_path", "source", "product_url", "image_url",
        "category_name", "brand_name",
    ]
    selected = selected[output_columns].reset_index(drop=True)
    chosen_ids = set(selected["product_id"])
    selected_reviews = reviews.loc[reviews["product_id"].isin(chosen_ids)].copy()
    selected_reviews["source"] = "Tiki"
    selected_reviews = selected_reviews.rename(columns={"created_at": "created_at_source"})
    selected_reviews["created_at"] = selected_reviews["created_at_source"].astype("string")
    selected_reviews = selected_reviews[
        ["review_id", "product_id", "rating", "title", "content", "created_at", "source"]
    ].reset_index(drop=True)
    selected_reviews["title"] = selected_reviews["title"].fillna("").astype(str)

    product_report = validate_products(selected, require_local_images=False)
    review_report = validate_reviews(selected_reviews, set(selected["product_id"]))
    product_report.raise_for_errors()
    review_report.raise_for_errors()
    return selected, selected_reviews


def write_staging(
    products: pd.DataFrame,
    reviews: pd.DataFrame,
    selection_manifest: Path | None = None,
) -> None:
    ensure_data_directories()
    products.to_parquet(PROCESSED_DIR / "products_staging.parquet", index=False)
    reviews.to_parquet(PROCESSED_DIR / "reviews_staging.parquet", index=False)
    manifest = {
        "kind": "prepared_tiki_snapshot",
        "rows": {"products": len(products), "reviews": len(reviews)},
        "categories": products["category"].value_counts().sort_index().to_dict(),
        "raw_sha256": {
            name: sha256(RAW_DIR / filename) for name, filename in SOURCE_FILES.items()
        },
    }
    if selection_manifest is not None:
        manifest["selection_manifest"] = str(selection_manifest.relative_to(PROJECT_ROOT)).replace("\\", "/")
        manifest["selection_manifest_sha256"] = sha256(selection_manifest)
        manifest["supplement_raw_sha256"] = {
            name: sha256(RAW_DIR / filename)
            for name, filename in SUPPLEMENT_FILES.items()
        }
    (PROCESSED_DIR / "dataset_staging_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chuẩn hóa snapshot Tiki thành schema Hợp Gu V2")
    parser.add_argument("--limit", type=int, default=60, help="Số sản phẩm staging cần chọn")
    parser.add_argument("--per-category", type=int, default=None, help="Giới hạn mỗi danh mục rộng")
    parser.add_argument(
        "--min-category-size", type=int, default=1,
        help="Chỉ giữ danh mục có ít nhất số candidate này",
    )
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=None,
        help="Manifest ID cố định để giữ catalog cũ và append coverage thật",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit < 1:
        raise ValueError("--limit phải lớn hơn 0")
    selection_manifest = args.selection_manifest
    if selection_manifest is not None and not selection_manifest.is_absolute():
        selection_manifest = PROJECT_ROOT / selection_manifest
    products, reviews = build_dataset(
        args.limit,
        args.per_category,
        args.min_category_size,
        selection_manifest,
    )
    write_staging(products, reviews, selection_manifest)
    print(f"Đã tạo staging: {len(products)} sản phẩm thật, {len(reviews)} review thật")
    print(products["category"].value_counts().sort_index().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

