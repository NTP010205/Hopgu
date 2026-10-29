from __future__ import annotations


def top_review_evidence(reviews: object, *, max_chars: int = 280) -> str:
    """Trả review thật đã xếp hữu ích trong pipeline, không sinh text mới."""
    if reviews is None or isinstance(reviews, (str, bytes, dict)):
        return ""
    try:
        iterator = iter(reviews)
    except TypeError:
        return ""
    for review in iterator:
        text = str(review).strip()
        if text:
            return text[:max_chars]
    return ""

