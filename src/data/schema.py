from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


PRODUCT_REQUIRED_COLUMNS = (
    "product_id",
    "product_name",
    "category",
    "price_vnd",
    "description",
    "rating",
    "review_count",
    "reviews_text",
    "image_path",
    "source",
    "product_url",
)

REVIEW_REQUIRED_COLUMNS = (
    "review_id",
    "product_id",
    "rating",
    "title",
    "content",
    "created_at",
    "source",
)


@dataclass(frozen=True, slots=True)
class ProductRecord:
    product_id: str
    product_name: str
    category: str
    price_vnd: int
    description: str
    rating: float
    review_count: int
    reviews_text: list[str] = field(default_factory=list)
    image_path: str = ""
    source: str = "Tiki"
    product_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ReviewRecord:
    review_id: str
    product_id: str
    rating: float
    title: str
    content: str
    created_at: str | None
    source: str = "Tiki"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

