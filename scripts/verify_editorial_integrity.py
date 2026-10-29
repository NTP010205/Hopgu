"""Verify protected AI/backend bytes while allowing presentation-only changes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parents[2] / "hopgu_v2"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import SETTINGS


BASELINE_PATH = ROOT / "integration_backup_20261025/checksums_before.json"
baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def digest(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


# The canonical backend and every protected target byte remain strict. UI/CSS is
# intentionally outside this set because responsive fixes are presentation-only.
protected_roots = (
    "src/",
    "data/processed/",
    "data/embeddings/",
    "data/images/",
    "data/eval/",
)
source_changed = [
    name for name, before in baseline["source"].items()
    if digest(SOURCE / name) != before
]
protected_names = [
    name for name in baseline["source"]
    if name.startswith(protected_roots)
]
target_protected_changed = [
    name for name in protected_names
    if digest(ROOT / name) != baseline["target"].get(name)
]
target_source_mismatch = [
    name for name in protected_names
    if digest(ROOT / name) != digest(SOURCE / name)
]
new_protected: list[str] = []
for folder in protected_roots:
    new_protected.extend(
        str(path.relative_to(ROOT)).replace("\\", "/")
        for path in (ROOT / folder).rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and str(path.relative_to(ROOT)).replace("\\", "/") not in baseline["target"]
    )

original_html = (ROOT / "integration_backup_20261025/hopgu_v2_editorial_frontend.html").read_text(encoding="utf-8")
current_html = (ROOT / "hopgu_v2_editorial_frontend.html").read_text(encoding="utf-8")
editorial_css_unchanged = (
    re.findall(r"<style>([\s\S]*?)</style>", original_html)
    == re.findall(r"<style>([\s\S]*?)</style>", current_html)
)

# Semantic contracts make the protected scope explicit in the report instead
# of weakening checksum coverage to accommodate the presentation delta.
products_count = len(pd.read_parquet(ROOT / "data/processed/products.parquet", columns=["product_id"]))
reviews_count = len(pd.read_parquet(ROOT / "data/processed/reviews.parquet"))
runtime_contract = {
    "products": products_count,
    "reviews": reviews_count,
    "text_model": SETTINGS.text_model,
    "vision_model": SETTINGS.vision_model,
    "ollama_model": SETTINGS.ollama_model,
    "fusion_alpha": SETTINGS.fusion_alpha,
}
runtime_contract_expected = {
    "products": 860,
    "reviews": 14219,
    "text_model": "intfloat/multilingual-e5-small",
    "vision_model": "openai/clip-vit-base-patch32",
    "ollama_model": "qwen2.5:3b-instruct",
    "fusion_alpha": 0.8,
}
runtime_contract_ok = runtime_contract == runtime_contract_expected

adapter_source = (ROOT / "serve_editorial.py").read_text(encoding="utf-8")
adapter_contract = {
    "uses_recommendation_service": "from src.recommender.service import RecommendationService" in adapter_source,
    "dispatches_text": "service.search_text(" in adapter_source,
    "dispatches_image": "service.search_image(" in adapter_source,
    "dispatches_multimodal": "service.search_multimodal(" in adapter_source,
    "validates_uploaded_image": "validate_uploaded_image(" in adapter_source,
}
adapter_contract_ok = all(adapter_contract.values())

protected_ok = not (
    source_changed
    or target_protected_changed
    or target_source_mismatch
    or new_protected
)
report = {
    "source_files_checked": len(baseline["source"]),
    "protected_files_checked": len(protected_names),
    "protected_roots": list(protected_roots),
    "source_changed": source_changed,
    "target_protected_changed": target_protected_changed,
    "target_source_mismatch": target_source_mismatch,
    "new_protected": new_protected,
    "protected_backend_and_data_ok": protected_ok,
    "runtime_contract": runtime_contract,
    "runtime_contract_expected": runtime_contract_expected,
    "runtime_contract_ok": runtime_contract_ok,
    "adapter_contract": adapter_contract,
    "adapter_contract_ok": adapter_contract_ok,
    "presentation_only": {
        "file": "hopgu_v2_editorial_frontend.html",
        "editorial_css_unchanged": editorial_css_unchanged,
        "changes_allowed": True,
        "affects_integrity_result": False,
    },
    "latest_evaluation_sha256": digest(ROOT / "data/eval/results/latest.json"),
}
report["ok"] = protected_ok and runtime_contract_ok and adapter_contract_ok

output = ROOT / "integration_verification/integrity.json"
output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
assert report["ok"], "Protected backend/data integrity or runtime contract changed"
