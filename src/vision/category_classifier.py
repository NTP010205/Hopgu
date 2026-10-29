from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from src.vision.image_encoder import ImageEncoder


# English visual descriptions work reliably with the pretrained CLIP checkpoint.
# They describe only the existing broad catalog taxonomy; no product data is added.
CATEGORY_PROMPTS: dict[str, tuple[str, ...]] = {
    "Bách hóa": (
        "a product photo of groceries, packaged food, milk, rice, snacks or cooking ingredients",
        "an online store photo of food, beverages and everyday grocery products",
    ),
    "Làm đẹp & Sức khỏe": (
        "a product photo of cosmetics, skincare, sunscreen, shampoo or personal care",
        "an online store photo of beauty, health and body care products",
    ),
    "Nhà cửa & Đời sống": (
        "a product photo of household goods, drinkware, a lunch box or kitchen utensils",
        "an online store photo of home, kitchen and everyday living products",
    ),
    "Sách": (
        "a product photo of a book, novel or printed publication with a book cover",
        "an online bookstore photo of a physical book",
    ),
    "Thời trang nam": (
        "a product photo of men's clothing, underwear, shirt, trousers or menswear",
        "an online fashion photo of men's clothes or men's shoes",
    ),
    "Thời trang nữ": (
        "a product photo of women's clothing, lingerie, dress, blouse or womenswear",
        "an online fashion photo of women's clothes, handbag or women's shoes",
    ),
    "Điện gia dụng": (
        "a product photo of an electric home appliance, rice cooker, blender or iron",
        "an online store photo of a kitchen or household electrical appliance",
    ),
    "Điện tử & Phụ kiện": (
        "a product photo of electronics, headphones, mouse, keyboard, speaker or phone accessory",
        "an online electronics store photo of a computer, audio or mobile accessory",
    ),
}


@dataclass(frozen=True, slots=True)
class CategoryPrediction:
    primary_category: str
    candidate_categories: tuple[str, ...]
    scores: dict[str, float]
    confidence: str
    margin: float
    strategy: str

    @property
    def uncertain(self) -> bool:
        return self.confidence != "high"

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["candidate_categories"] = list(self.candidate_categories)
        payload["uncertain"] = self.uncertain
        return payload


class ZeroShotCategoryClassifier:
    """Broad-category inference from CLIP image/text similarity.

    Confidence thresholds are fixed engineering guardrails, not fitted on the
    evaluation labels. A weak prediction widens the candidate pool instead of
    pretending the category is certain.
    """

    HIGH_MARGIN = 0.015
    USABLE_MARGIN = 0.004
    MIN_SCORE = 0.16

    def __init__(self, image_encoder: ImageEncoder, categories: list[str]) -> None:
        self.image_encoder = image_encoder
        self.categories = [category for category in categories if category in CATEGORY_PROMPTS]
        self._prototypes: np.ndarray | None = None

    @property
    def prototypes(self) -> np.ndarray:
        if self._prototypes is None:
            prompt_groups = [CATEGORY_PROMPTS[category] for category in self.categories]
            flat_prompts = [prompt for group in prompt_groups for prompt in group]
            encoded = self.image_encoder.encode_texts(flat_prompts)
            cursor = 0
            prototypes: list[np.ndarray] = []
            for group in prompt_groups:
                group_vectors = encoded[cursor : cursor + len(group)]
                cursor += len(group)
                prototype = group_vectors.mean(axis=0)
                prototype /= max(float(np.linalg.norm(prototype)), 1e-12)
                prototypes.append(prototype)
            self._prototypes = np.asarray(prototypes, dtype=np.float32)
        return self._prototypes

    def predict_vector(self, image_vector: np.ndarray) -> CategoryPrediction:
        if not self.categories:
            raise ValueError("Catalog không có danh mục hỗ trợ CLIP zero-shot")
        vector = np.asarray(image_vector, dtype=np.float32).reshape(-1)
        vector /= max(float(np.linalg.norm(vector)), 1e-12)
        similarities = self.prototypes @ vector
        order = np.argsort(similarities)[::-1]
        first = int(order[0])
        second = int(order[1]) if len(order) > 1 else first
        top_score = float(similarities[first])
        margin = top_score - float(similarities[second])

        if top_score >= self.MIN_SCORE and margin >= self.HIGH_MARGIN:
            confidence = "high"
            candidates = (self.categories[first],)
            strategy = "top_1_category"
        elif top_score >= self.MIN_SCORE and margin >= self.USABLE_MARGIN:
            confidence = "medium"
            candidates = (self.categories[first], self.categories[second])
            strategy = "top_2_categories"
        else:
            confidence = "low"
            candidates = ()
            strategy = "global_catalog"

        return CategoryPrediction(
            primary_category=self.categories[first],
            candidate_categories=candidates,
            scores={
                self.categories[index]: float(similarities[index])
                for index in order
            },
            confidence=confidence,
            margin=margin,
            strategy=strategy,
        )

