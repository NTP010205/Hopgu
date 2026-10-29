#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Thin HTTP adapter for the Hợp Gu editorial frontend.

Recommendation logic remains exclusively in ``RecommendationService``. This
module parses HTTP input, validates uploads with the existing vision helper,
calls the service, and serializes real catalog/result fields for the browser.
"""

from __future__ import annotations

import argparse
import json
import math
import mimetypes
import threading
import webbrowser
from dataclasses import asdict
from email.parser import BytesParser
from email.policy import default as email_policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse

import numpy as np
import pandas as pd

from src.config import EMBEDDINGS_DIR, EVAL_RESULTS_DIR, IMAGES_DIR, PROJECT_ROOT, SETTINGS
from src.data.loader import load_products
from src.nlp.llm_service import OllamaService
from src.nlp.review_analyzer import top_review_evidence
from src.recommender.service import RecommendationService
from src.vision.image_encoder import MAX_UPLOAD_BYTES, ImageValidationError, validate_uploaded_image


FRONTEND_FILE = PROJECT_ROOT / "hopgu_v2_editorial_frontend.html"
MAX_REQUEST_BYTES = MAX_UPLOAD_BYTES + 1024 * 1024
MAX_TOP_K = 40

_service: RecommendationService | None = None
_service_lock = threading.RLock()
_catalog: pd.DataFrame | None = None
_catalog_lock = threading.Lock()
_status_client = OllamaService()


def _json_value(value: Any) -> Any:
    """Convert pandas/numpy values into strict JSON-compatible values."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and missing:
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (str, int)):
        return value
    return str(value)


def _get_catalog() -> pd.DataFrame:
    global _catalog
    if _catalog is None:
        with _catalog_lock:
            if _catalog is None:
                _catalog = load_products().reset_index(drop=True)
    return _catalog


def _get_service() -> RecommendationService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = RecommendationService(products=_get_catalog())
    return _service


def _local_image_url(image_path: Any) -> str | None:
    if image_path is None:
        return None
    filename = Path(str(image_path)).name
    if not filename:
        return None
    candidate = (IMAGES_DIR / filename).resolve()
    image_root = IMAGES_DIR.resolve()
    try:
        candidate.relative_to(image_root)
    except ValueError:
        return None
    if not candidate.is_file():
        return None
    return f"/data/images/{quote(filename)}"


def _catalog_record(row: pd.Series) -> dict[str, Any]:
    fields = (
        "product_id",
        "product_name",
        "category",
        "category_name",
        "price_vnd",
        "rating",
        "review_count",
        "description",
        "brand_name",
        "source",
        "product_url",
    )
    record = {field: _json_value(row[field]) for field in fields if field in row.index}
    record["image_url"] = _local_image_url(row.get("image_path"))
    # Both renderer keys use the same safe local endpoint. Product images never
    # fall back to an upstream/demo URL.
    record["image_uri"] = record["image_url"]
    evidence = top_review_evidence(row.get("reviews_text"))
    record["review_evidence"] = [evidence] if evidence else []
    return record


def _result_records(results: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for rank, (_, row) in enumerate(results.iterrows(), start=1):
        record = _catalog_record(row)
        record["rank"] = rank
        for field in ("text_score", "image_score", "final_score", "reason"):
            record[field] = _json_value(row[field]) if field in row.index else None
        records.append(record)
    return records


def _artifact_status() -> dict[str, Any]:
    required = {
        "catalog": PROJECT_ROOT / "data" / "processed" / "products.parquet",
        "text_embeddings": EMBEDDINGS_DIR / "text_embeddings.npy",
        "text_manifest": EMBEDDINGS_DIR / "text_manifest.json",
        "image_embeddings": EMBEDDINGS_DIR / "image_embeddings.npy",
        "image_manifest": EMBEDDINGS_DIR / "image_manifest.json",
        "evaluation": EVAL_RESULTS_DIR / "latest.json",
    }
    return {
        name: {"ready": path.is_file(), "file": path.name}
        for name, path in required.items()
    }


def backend_status_payload() -> dict[str, Any]:
    artifacts = _artifact_status()
    missing = [name for name, info in artifacts.items() if not info["ready"]]
    catalog_count: int | None = None
    catalog_error: str | None = None
    try:
        catalog_count = len(_get_catalog())
    except Exception as exc:  # report loader failures instead of hiding them
        catalog_error = f"{type(exc).__name__}: {exc}"

    llm_status = _status_client.status()
    state = "ERROR" if missing or catalog_error else ("READY" if llm_status.ready else "FALLBACK")
    return {
        "status": "ok" if state != "ERROR" else "error",
        "backend_status": state,
        "catalog_count": catalog_count,
        "catalog_error": catalog_error,
        "artifacts": artifacts,
        "models": {
            "text": SETTINGS.text_model,
            "vision": SETTINGS.vision_model,
            "ollama": SETTINGS.ollama_model,
        },
        "fusion_alpha": SETTINGS.fusion_alpha,
        "llm": asdict(llm_status),
    }


def _optional_int(value: str | None, field: str) -> int | None:
    if value is None or not value.strip():
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{field} phải là số nguyên") from exc
    if parsed < 0:
        raise ValueError(f"{field} không được âm")
    return parsed


def _parse_multipart(content_type: str, body: bytes) -> tuple[dict[str, str], dict[str, Any] | None]:
    message = BytesParser(policy=email_policy).parsebytes(
        (f"Content-Type: {content_type}\r\n" "MIME-Version: 1.0\r\n\r\n").encode("ascii")
        + body
    )
    if not message.is_multipart():
        raise ValueError("Request phải dùng multipart/form-data")

    fields: dict[str, str] = {}
    image_part: dict[str, Any] | None = None
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        payload = part.get_payload(decode=True) or b""
        filename = part.get_filename()
        if name == "image" and filename:
            if image_part is not None:
                raise ValueError("Chỉ chấp nhận một file ảnh")
            image_part = {
                "filename": Path(filename).name,
                "content_type": part.get_content_type(),
                "content": payload,
            }
        elif filename is None:
            charset = part.get_content_charset() or "utf-8"
            try:
                fields[name] = payload.decode(charset).strip()
            except (LookupError, UnicodeDecodeError) as exc:
                raise ValueError(f"Field {name} không phải UTF-8 hợp lệ") from exc
    return fields, image_part


class HopGuServer(ThreadingHTTPServer):
    daemon_threads = True


class HopGuRequestHandler(BaseHTTPRequestHandler):
    """HTTP boundary only; retrieval/ranking remains in RecommendationService."""

    server_version = "HopGuEditorial/2"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[HTTP] {self.address_string()} {fmt % args}", flush=True)

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(_json_value(payload), ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, code: str, message: str) -> None:
        self._send_json(
            {"status": "error", "error": {"code": code, "message": message}},
            status=status,
        )

    def _send_html(self) -> None:
        if not FRONTEND_FILE.is_file():
            self._error(500, "FRONTEND_MISSING", f"Không tìm thấy {FRONTEND_FILE.name}")
            return
        body = FRONTEND_FILE.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _send_image(self, route: str) -> None:
        filename = unquote(route.removeprefix("/data/images/"))
        if not filename or Path(filename).name != filename:
            self._error(400, "INVALID_IMAGE_PATH", "Đường dẫn ảnh không hợp lệ")
            return
        image_root = IMAGES_DIR.resolve()
        candidate = (image_root / filename).resolve()
        try:
            candidate.relative_to(image_root)
        except ValueError:
            self._error(400, "INVALID_IMAGE_PATH", "Đường dẫn ảnh không hợp lệ")
            return
        if not candidate.is_file():
            self._error(404, "IMAGE_NOT_FOUND", "Không tìm thấy ảnh sản phẩm")
            return
        mime = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        if mime not in {"image/jpeg", "image/png", "image/webp"}:
            self._error(415, "UNSUPPORTED_IMAGE", "Định dạng ảnh không được hỗ trợ")
            return
        content = candidate.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "public, max-age=86400")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:
        route = urlparse(self.path).path
        if route in {"/", "/index.html", f"/{FRONTEND_FILE.name}", "/hopgu_standalone_full.html"}:
            self._send_html()
            return
        if route == "/api/status":
            payload = backend_status_payload()
            self._send_json(payload, 200 if payload["backend_status"] != "ERROR" else 503)
            return
        if route == "/api/catalog":
            try:
                products = [_catalog_record(row) for _, row in _get_catalog().iterrows()]
            except Exception as exc:
                self._error(503, "CATALOG_UNAVAILABLE", f"Không thể nạp catalog thật: {exc}")
                return
            self._send_json({"status": "ok", "count": len(products), "products": products})
            return
        if route.startswith("/data/images/"):
            self._send_image(route)
            return
        self._error(404, "NOT_FOUND", "Không tìm thấy đường dẫn")

    def do_POST(self) -> None:
        route = urlparse(self.path).path
        if route != "/api/recommend":
            self._error(404, "NOT_FOUND", "Không tìm thấy đường dẫn")
            return

        content_type = self.headers.get("Content-Type", "")
        if not content_type.lower().startswith("multipart/form-data"):
            self._error(415, "UNSUPPORTED_MEDIA_TYPE", "Hãy gửi multipart/form-data")
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._error(400, "INVALID_CONTENT_LENGTH", "Content-Length không hợp lệ")
            return
        if content_length <= 0:
            self._error(400, "EMPTY_REQUEST", "Request rỗng")
            return
        if content_length > MAX_REQUEST_BYTES:
            self._error(413, "REQUEST_TOO_LARGE", "Request vượt giới hạn cho phép")
            return

        image = None
        try:
            fields, image_part = _parse_multipart(content_type, self.rfile.read(content_length))
            query = fields.get("query", "").strip()
            category = fields.get("category") or None
            price_min = _optional_int(fields.get("min_price"), "min_price")
            price_max = _optional_int(fields.get("max_price"), "max_price")
            top_k = _optional_int(fields.get("top_k"), "top_k") or 20
            if not 1 <= top_k <= MAX_TOP_K:
                raise ValueError(f"top_k phải nằm trong [1, {MAX_TOP_K}]")
            if price_min is not None and price_max is not None and price_min > price_max:
                raise ValueError("min_price không được lớn hơn max_price")
            if category and category not in set(_get_catalog()["category"].astype(str)):
                raise ValueError("Danh mục không tồn tại trong catalog")
            if not query and image_part is None:
                raise ValueError("Cần nhập query hoặc chọn một ảnh")

            if image_part is not None:
                image = validate_uploaded_image(
                    image_part["content"],
                    filename=image_part["filename"],
                )

            service = _get_service()
            with _service_lock:
                if image is None:
                    mode = "TEXT"
                    results = service.search_text(
                        query,
                        category=category,
                        price_min=price_min,
                        price_max=price_max,
                        top_k=top_k,
                    )
                    metadata = dict(results.attrs.get("search_metadata", {}))
                elif not query:
                    mode = "IMAGE"
                    results = service.search_image(
                        image,
                        category=category,
                        price_min=price_min,
                        price_max=price_max,
                        top_k=top_k,
                    )
                    metadata = dict(results.attrs.get("search_metadata", {}))
                else:
                    mode = "MULTIMODAL"
                    # Do not pass alpha. The service owns its configured fusion
                    # value (0.8 in the source-of-truth backend).
                    results, metadata = service.search_multimodal(
                        query,
                        image,
                        category=category,
                        price_min=price_min,
                        price_max=price_max,
                        top_k=top_k,
                    )

            llm_status = service.llm.status()
            backend_state = "READY" if llm_status.ready else "FALLBACK"
            self._send_json(
                {
                    "status": "ok",
                    "mode": mode,
                    "backend_status": backend_state,
                    "query": query,
                    "count": len(results),
                    "results": _result_records(results),
                    "metadata": metadata,
                    "llm": asdict(llm_status),
                }
            )
        except ImageValidationError as exc:
            self._error(400, "INVALID_IMAGE", str(exc))
        except ValueError as exc:
            self._error(400, "INVALID_REQUEST", str(exc))
        except (FileNotFoundError, RuntimeError) as exc:
            self._error(503, "BACKEND_UNAVAILABLE", f"Không thể chạy hệ thống gợi ý: {exc}")
        except Exception as exc:
            self._error(500, "RECOMMENDATION_ERROR", f"Hệ thống gợi ý gặp lỗi: {type(exc).__name__}: {exc}")
        finally:
            if image is not None:
                image.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Hợp Gu Editorial API adapter")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--open", action="store_true", help="Mở trình duyệt sau khi server khởi động")
    args = parser.parse_args()

    if not FRONTEND_FILE.is_file():
        raise FileNotFoundError(f"Không tìm thấy frontend: {FRONTEND_FILE}")

    status = backend_status_payload()
    url = f"http://{args.host}:{args.port}"
    print("=" * 68)
    print("✦ HỢP GU EDITORIAL — REAL RECOMMENDATION API")
    print(f"✦ URL: {url}")
    print(f"✦ Backend: {status['backend_status']} · Catalog: {status['catalog_count']}")
    print(f"✦ Fusion alpha (backend-owned): {SETTINGS.fusion_alpha}")
    print("✦ Endpoints: GET /api/status, GET /api/catalog, POST /api/recommend")
    print("=" * 68, flush=True)

    server = HopGuServer((args.host, args.port), HopGuRequestHandler)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Hợp Gu] Đã dừng máy chủ.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
