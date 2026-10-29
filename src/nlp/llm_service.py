from __future__ import annotations

import json
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import requests

from src.config import SETTINGS


@dataclass(frozen=True, slots=True)
class LLMStatus:
    ready: bool
    mode: str
    detail: str


class OllamaService:
    """Client nhỏ, timeout hữu hạn; Ollama chỉ sinh JSON/evidence-grounded text."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        *,
        timeout: float = 3.0,
    ) -> None:
        self.base_url = (base_url or SETTINGS.ollama_url).rstrip("/")
        self.model = model or SETTINGS.ollama_model
        self.timeout = timeout
        self._cached_status: LLMStatus | None = None
        self._status_checked_at = 0.0
        self._parse_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._reason_cache: OrderedDict[str, str] = OrderedDict()

    @staticmethod
    def _cache_put(cache: OrderedDict, key: str, value: Any, limit: int) -> None:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > limit:
            cache.popitem(last=False)

    def status(self) -> LLMStatus:
        now = time.monotonic()
        if self._cached_status is not None and now - self._status_checked_at < 5.0:
            return self._cached_status
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=min(self.timeout, 1.5))
            response.raise_for_status()
            models = response.json().get("models", [])
            names = {str(item.get("name", "")) for item in models}
            installed = self.model in names or any(
                name.split(":", 1)[0] == self.model.split(":", 1)[0] for name in names
            )
            if not installed:
                result = LLMStatus(False, "fallback", f"Ollama chạy nhưng chưa có model {self.model}")
            else:
                result = LLMStatus(True, "ollama", f"Ollama sẵn sàng với {self.model}")
        except (requests.RequestException, ValueError, TypeError) as exc:
            result = LLMStatus(False, "fallback", f"Không kết nối được Ollama: {type(exc).__name__}")
        self._cached_status = result
        self._status_checked_at = now
        return result

    def _chat_json(self, system: str, user: str) -> Any:
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "format": "json",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "options": {"temperature": 0},
            },
            timeout=max(self.timeout, 30.0),
        )
        response.raise_for_status()
        content = response.json()["message"]["content"]
        return json.loads(content)

    def parse_query(self, query: str, allowed_categories: list[str]) -> dict[str, Any]:
        cache_key = json.dumps(
            {"model": self.model, "query": query, "allowed_categories": allowed_categories},
            ensure_ascii=False,
            sort_keys=True,
        )
        if cache_key in self._parse_cache:
            self._parse_cache.move_to_end(cache_key)
            return dict(self._parse_cache[cache_key])
        schema = {
            "intent": "search_product",
            "category": "một giá trị trong allowed_categories hoặc null",
            "colors": ["tên màu cơ bản xuất hiện nguyên văn trong query, không kèm từ 'màu'"],
            "attributes": ["thuộc tính sản phẩm xuất hiện trong query; không đưa giới tính hoặc điều kiện giá vào đây"],
            "price_min": "số nguyên VND chỉ khi query nói trên/từ/tối thiểu; nếu không thì null",
            "price_max": "số nguyên VND chỉ khi query nói dưới/không quá/tối đa/rẻ hơn; nếu không thì null",
            "gender": "nam, nữ, unisex hoặc null",
            "keywords": ["từ khóa nguyên văn quan trọng"],
        }
        result = self._chat_json(
            "Bạn là bộ phân tích truy vấn mua sắm tiếng Việt. Chỉ trích xuất thông tin có trong câu; "
            "không bịa thuộc tính hay cận giá. 'Dưới/rẻ hơn 500 nghìn' nghĩa là "
            "price_min=null, price_max=500000; 'trên 1 triệu' nghĩa là "
            "price_min=1000000, price_max=null. Giữ đủ tám key trong schema. "
            "Trả đúng một JSON object, không markdown.",
            json.dumps(
                {"query": query, "allowed_categories": allowed_categories, "schema": schema},
                ensure_ascii=False,
            ),
        )
        if isinstance(result, dict):
            self._cache_put(self._parse_cache, cache_key, dict(result), 256)
        return result

    def generate_reasons(self, query: str, evidence: list[dict[str, Any]]) -> dict[str, str]:
        reasons: dict[str, str] = {}
        system = (
            "Bạn tạo một lý do có cấu trúc cho đúng một sản phẩm. Chỉ dùng đúng các giá trị evidence; "
            "không suy diễn, đánh giá chủ quan hoặc thêm thuộc tính. Phải giữ nguyên điểm, giá và rating. "
            "Nếu có review, phần trong dấu nháy phải là một đoạn 5-15 từ liên tiếp được chép nguyên văn "
            "từ evidence.review, không được diễn giải. Trả JSON object có đúng một key là product_id "
            "được yêu cầu, ánh xạ tới reason theo đúng required_reason_format; không thêm key khác."
        )
        for item in evidence:
            product_id = str(item["product_id"])
            cache_key = json.dumps(
                {"model": self.model, "query": query, "evidence": item},
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
            if cache_key in self._reason_cache:
                self._reason_cache.move_to_end(cache_key)
                reasons[product_id] = self._reason_cache[cache_key]
                continue
            score_text = f"{float(item['recommendation_score']):.4f}"
            rating_text = f"{float(item['rating']):g}"
            prefix = (
                f"Điểm gợi ý {score_text}; giá {int(item['price_vnd'])} VND; "
                f"rating {rating_text}/5"
            )
            review = str(item.get("review", "")).strip()
            required_review_quote = ""
            if review:
                words = list(re.finditer(r"\S+", review))
                quote_end = words[min(4, len(words) - 1)].end()
                required_review_quote = review[:quote_end].rstrip(".,…!?;:")
                required_format = (
                    prefix + f'; review thật: "{required_review_quote}".'
                )
            else:
                required_format = (
                    prefix
                    + f"; sản phẩm {item['name']} thuộc danh mục {item['category']}."
                )
            request = {
                "query": query,
                "required_product_id": product_id,
                "evidence": item,
                "required_reason_format": required_format,
                "required_review_quote": required_review_quote or None,
                "required_output_example": {product_id: required_format},
            }
            reason = ""
            for attempt in range(2):
                attempt_system = system
                if attempt:
                    attempt_system += (
                        " Lần trước sai format. Bắt buộc trả đúng object theo required_output_example, "
                        "giữ nguyên key product_id và thay value bằng reason."
                    )
                payload = self._chat_json(
                    attempt_system,
                    json.dumps(request, ensure_ascii=False),
                )
                if isinstance(payload, dict):
                    current = {
                        str(key): str(value).strip() for key, value in payload.items()
                    }
                    if set(current) == {product_id} and current[product_id]:
                        candidate = current[product_id]
                        if review:
                            reason_prefix = prefix + '; review thật: "'
                            quote = ""
                            if candidate.startswith(reason_prefix):
                                tail = candidate[len(reason_prefix):]
                                if tail.endswith('".'):
                                    quote = tail[:-2]
                                elif tail.endswith('"'):
                                    quote = tail[:-1]
                            if quote and quote == required_review_quote:
                                reason = candidate
                                break
                        elif candidate == required_format:
                            reason = candidate
                            break
            if not reason:
                raise ValueError("LLM reason response không khớp product_id evidence")
            reasons[product_id] = reason
            self._cache_put(self._reason_cache, cache_key, reason, 1024)
        return reasons

