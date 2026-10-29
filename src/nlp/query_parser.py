from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any

from src.nlp.llm_service import OllamaService


DEFAULT_CATEGORIES = [
    "Bách hóa",
    "Làm đẹp & Sức khỏe",
    "Nhà cửa & Đời sống",
    "Sách",
    "Thể thao & Dã ngoại",
    "Thời trang nam",
    "Thời trang nữ",
    "Điện gia dụng",
    "Điện tử & Phụ kiện",
]


@dataclass(slots=True)
class ParsedQuery:
    intent: str = "search_product"
    category: str | None = None
    category_candidates: list[str] = field(default_factory=list)
    category_confidence: str = "none"
    category_alias: str | None = None
    colors: list[str] = field(default_factory=list)
    attributes: list[str] = field(default_factory=list)
    price_min: int | None = None
    price_max: int | None = None
    gender: str | None = None
    keywords: list[str] = field(default_factory=list)
    parser_mode: str = "fallback"
    parser_detail: str = "rule-based"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _fold(text: str) -> str:
    value = unicodedata.normalize("NFD", text.lower().replace("đ", "d"))
    return "".join(char for char in value if unicodedata.category(char) != "Mn")


def _contains_phrase(text: str, phrase: str) -> bool:
    folded_text = _fold(text)
    folded_phrase = _fold(phrase).strip()
    if not folded_phrase:
        return False
    return re.search(
        rf"(?<![a-z0-9]){re.escape(folded_phrase)}(?![a-z0-9])", folded_text
    ) is not None


def _contains_color(text: str, color: str) -> bool:
    direct = re.search(
        rf"(?<!\w){re.escape(color.lower())}(?!\w)", text.lower()
    ) is not None
    folded = _fold(text)
    after_color_word = re.search(
        rf"(?<![a-z0-9])mau\s+{re.escape(_fold(color))}(?![a-z0-9])", folded
    ) is not None
    return direct or after_color_word


def _money_to_vnd(number: str, unit: str | None) -> int:
    value = float(number.replace(",", "."))
    folded_unit = _fold(unit or "")
    if "tr" in folded_unit or "trieu" in folded_unit:
        value *= 1_000_000
    elif "nghin" in folded_unit or "ngan" in folded_unit or "k" == folded_unit.strip():
        value *= 1_000
    return int(value)


def _rule_price(query: str) -> tuple[int | None, int | None]:
    folded = _fold(query)
    amount = r"(\d+(?:[\.,]\d+)?)\s*(trieu|tr|nghin|ngan|k)?"
    range_match = re.search(rf"(?:tu)\s*{amount}\s*(?:den|toi|-)\s*{amount}", folded)
    if range_match:
        return (
            _money_to_vnd(range_match.group(1), range_match.group(2)),
            _money_to_vnd(range_match.group(3), range_match.group(4)),
        )
    max_match = re.search(
        rf"(?:duoi|khong qua|toi da|nho hon|re hon)\s*{amount}", folded
    )
    min_match = re.search(rf"(?:tren|toi thieu|lon hon)\s*{amount}", folded)
    return (
        _money_to_vnd(min_match.group(1), min_match.group(2)) if min_match else None,
        _money_to_vnd(max_match.group(1), max_match.group(2)) if max_match else None,
    )


CATEGORY_HINTS = (
    ("Điện tử & Phụ kiện", ("bluetooth", "laptop", "dien thoai", "chuot", "ban phim", "loa vi tinh")),
    ("Làm đẹp & Sức khỏe", ("da dau", "sua rua mat", "dau goi", "kem duong")),
    ("Thể thao & Dã ngoại", ("chay bo", "the thao", "da ngoai", "gym")),
    ("Thời trang nữ", ("ao nu", "quan nu", "vay nu", "dam nu")),
    ("Thời trang nam", ("ao nam", "quan nam", "do nam")),
    ("Sách", ("sach", "truyen", "tieu thuyet")),
    ("Điện gia dụng", ("may xay", "noi com", "ban ui", "quat", "may say", "may loc")),
    ("Bách hóa", ("ca phe", "sua", "banh", "keo", "thuc pham")),
    ("Nhà cửa & Đời sống", ("binh giu nhiet", "nha cua", "do dung bep")),
)

COLORS = ("đen", "trắng", "đỏ", "xanh", "vàng", "hồng", "tím", "nâu", "xám", "bạc")
ATTRIBUTES = (
    "đeo chéo", "nhẹ", "chạy bộ", "bluetooth", "chống ồn", "da dầu",
    "chống nắng", "chống nước", "laptop", "không dây", "giữ nhiệt",
)


class QueryParser:
    def __init__(
        self,
        llm: OllamaService | None = None,
        *,
        allowed_categories: list[str] | None = None,
    ) -> None:
        self.llm = llm or OllamaService()
        self.allowed_categories = allowed_categories or DEFAULT_CATEGORIES

    def parse(self, query: str) -> ParsedQuery:
        if not query.strip():
            raise ValueError("Query không được rỗng")
        status = self.llm.status()
        if status.ready:
            try:
                payload = self.llm.parse_query(query, self.allowed_categories)
                return self._from_llm(payload, status.detail, query)
            except Exception as exc:
                return self._rule_parse(query, f"LLM lỗi {type(exc).__name__}; dùng rule-based")
        return self._rule_parse(query, status.detail)

    def _from_llm(self, payload: dict[str, Any], detail: str, query: str) -> ParsedQuery:
        if not isinstance(payload, dict):
            raise ValueError("LLM parser response không phải object")
        rule_result = self._rule_parse(query, detail)
        category = rule_result.category
        category_candidates = list(rule_result.category_candidates)
        category_confidence = rule_result.category_confidence
        category_alias = rule_result.category_alias
        llm_category = payload.get("category")
        if (
            category is None
            and llm_category in self.allowed_categories
            and _contains_phrase(query, str(llm_category))
        ):
            category = str(llm_category)
            category_candidates = [category]
            category_confidence = "high"
            category_alias = "llm_explicit_category"

        raw_attributes = payload.get("attributes", [])
        if not isinstance(raw_attributes, list):
            raw_attributes = []
        llm_attributes = []
        for item in raw_attributes:
            value = str(item).strip()
            folded_value = _fold(value)
            is_price_or_gender = (
                any(char.isdigit() for char in value)
                or folded_value in {"nam", "nu", "unisex"}
                or any(
                    marker in folded_value
                    for marker in ("duoi ", "tren ", "toi da", "toi thieu", "re hon")
                )
            )
            if value and not is_price_or_gender and _contains_phrase(query, value):
                llm_attributes.append(value)

        raw_keywords = payload.get("keywords", [])
        if not isinstance(raw_keywords, list):
            raw_keywords = []
        llm_keywords = [
            str(item).strip()
            for item in raw_keywords
            if str(item).strip() and _contains_phrase(query, str(item))
        ]

        attributes = list(dict.fromkeys([*llm_attributes, *rule_result.attributes]))
        keywords = llm_keywords or rule_result.keywords
        return ParsedQuery(
            intent="search_product",
            category=category,
            category_candidates=category_candidates,
            category_confidence=category_confidence,
            category_alias=category_alias,
            colors=rule_result.colors,
            attributes=attributes,
            price_min=rule_result.price_min,
            price_max=rule_result.price_max,
            gender=rule_result.gender,
            keywords=keywords,
            parser_mode="ollama",
            parser_detail=detail,
        )

    def _rule_parse(self, query: str, detail: str) -> ParsedQuery:
        folded = _fold(query)
        gender = (
            "nữ" if " nu " in f" {folded} "
            else "nam" if " nam " in f" {folded} "
            else "unisex" if " unisex " in f" {folded} "
            else None
        )
        category = None
        category_candidates: list[str] = []
        category_confidence = "none"
        category_alias = None

        def choose(label: str, alias: str) -> None:
            nonlocal category, category_candidates, category_confidence, category_alias
            if label in self.allowed_categories:
                category = label
                category_candidates = [label]
                category_confidence = "high"
                category_alias = alias

        if any(value in folded for value in ("may pha ca phe", "may pha espresso")):
            choose("Điện gia dụng", "máy pha cà phê/espresso")
        elif "bao cao su" in folded:
            choose("Làm đẹp & Sức khỏe", "bao cao su")
        elif any(value in folded for value in ("tui xach", "tui deo cheo", "handbag", "crossbody")):
            choose("Thời trang nữ", "túi xách/túi đeo chéo")
        elif any(value in folded for value in ("balo", "ba lo", "backpack")):
            choose("Điện tử & Phụ kiện", "balo/backpack")
        elif any(value in folded for value in ("tai nghe", "headphone", "headphones")):
            choose("Điện tử & Phụ kiện", "tai nghe/headphone")
        elif any(value in folded for value in ("kem chong nang", "my pham", "cosmetic")):
            choose("Làm đẹp & Sức khỏe", "kem chống nắng/mỹ phẩm")
        elif "giay" in folded or "sneaker" in folded:
            if "chay bo" in folded and "Thể thao & Dã ngoại" in self.allowed_categories:
                choose("Thể thao & Dã ngoại", "giày chạy bộ")
            elif gender == "nữ":
                choose("Thời trang nữ", "giày/sneaker nữ")
            elif gender == "nam":
                choose("Thời trang nam", "giày/sneaker nam")
            else:
                category_candidates = [
                    label
                    for label in ("Thời trang nam", "Thời trang nữ")
                    if label in self.allowed_categories
                ]
                category = category_candidates[0] if category_candidates else None
                category_confidence = "medium" if category_candidates else "none"
                category_alias = "giày/sneaker không rõ giới tính"

        if category is None and not category_candidates:
            for label, hints in CATEGORY_HINTS:
                if label in self.allowed_categories and any(hint in folded for hint in hints):
                    choose(label, next(hint for hint in hints if hint in folded))
                    break
        if category is None and gender == "nữ":
            choose("Thời trang nữ", "giới tính nữ")
        elif category is None and gender == "nam" and any(word in folded for word in ("giay", "ao", "quan")):
            choose("Thời trang nam", "giới tính nam")
        price_min, price_max = _rule_price(query)
        colors = [color for color in COLORS if _contains_color(query, color)]
        attributes = [attribute for attribute in ATTRIBUTES if _contains_phrase(query, attribute)]
        stop = {"tim", "san", "pham", "cho", "toi", "duoi", "tren", "mau", "nam", "nu"}
        keywords = [token for token in re.findall(r"[a-z0-9]+", folded) if len(token) > 1 and token not in stop]
        return ParsedQuery(
            category=category,
            category_candidates=category_candidates,
            category_confidence=category_confidence,
            category_alias=category_alias,
            colors=colors,
            attributes=attributes,
            price_min=price_min,
            price_max=price_max,
            gender=gender,
            keywords=keywords,
            parser_mode="fallback",
            parser_detail=detail,
        )

