#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HỢP GU ATELIER — Máy chủ tích hợp & Bộ đóng gói giao diện đầy đủ.

File này giúp bạn:
1. Tự động đọc toàn bộ dữ liệu sản phẩm thật từ thư mục dự án (products_catalog.json hoặc products.parquet)
   và nhúng trực tiếp vào giao diện Editorial Lookbook để tạo ra bản đóng gói độc lập
   `hopgu_standalone_full.html` (chỉ cần nhấp đúp chuột là mở được ở mọi máy tính).
2. Khởi chạy máy chủ web tại http://127.0.0.1:8000 cung cấp giao diện, phục vụ ảnh local và các API:
   - GET  /             : Giao diện Hợp Gu Atelier đầy đủ
   - GET  /api/catalog  : Trả về danh sách 860 sản phẩm thật dạng JSON
   - GET  /api/search   : Tìm kiếm & lọc sản phẩm theo từ khóa (tiếng Việt có dấu / không dấu), danh mục, mức giá
   - POST /api/checkout : Nhận đơn đặt hàng từ giỏ hàng trên web
   - GET  /data/images/ : Phục vụ toàn bộ ảnh sản phẩm local offline tốc độ cao
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

FRONTEND_FILE = "hopgu_v2_editorial_frontend.html"
ALT_FRONTEND_FILE = "h_p_gu_th_i_trang_thi_t_k_lookbook_cao_c_p.html"
STANDALONE_OUTPUT_FILE = "hopgu_standalone_full.html"


def ascii_fold(text: str) -> str:
    """Loại bỏ dấu tiếng Việt để tìm kiếm không phụ thuộc dấu."""
    t = unicodedata.normalize("NFD", str(text or "").lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    t = t.replace("đ", "d").replace("Đ", "d")
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def find_frontend_path(base_dir: Path) -> Path:
    candidates = [
        base_dir / FRONTEND_FILE,
        base_dir / ALT_FRONTEND_FILE,
        base_dir / "hopgu_v2" / FRONTEND_FILE,
        base_dir / "hopgu_v2" / ALT_FRONTEND_FILE,
        base_dir.parent / FRONTEND_FILE,
        base_dir.parent / "hopgu_v2" / FRONTEND_FILE,
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(f"Không tìm thấy file giao diện ({FRONTEND_FILE} hoặc {ALT_FRONTEND_FILE})")


def load_local_catalog_if_exists(base_dir: Path) -> List[Dict[str, Any]]:
    """
    Tìm và nạp toàn bộ dữ liệu sản phẩm từ các đường dẫn trong dự án.
    Nếu có products_catalog.json nạp ngay; nếu chỉ có products.parquet thì tự động chuyển đổi.
    """
    candidate_paths = [
        base_dir / "data" / "processed" / "products_catalog.json",
        base_dir / "hopgu_v2" / "data" / "processed" / "products_catalog.json",
        base_dir.parent / "hopgu_v2" / "data" / "processed" / "products_catalog.json",
        base_dir / "data" / "products.json",
        base_dir / "artifacts" / "catalog.json",
        base_dir / "hopgu_catalog_data.json",
    ]
    for path in candidate_paths:
        if path.exists() and path.is_file():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, list) and len(raw) > 0:
                    print(f"[Hợp Gu] Đã nạp {len(raw)} sản phẩm từ: {path}")
                    return raw
                if isinstance(raw, dict) and isinstance(raw.get("products"), list):
                    products = raw["products"]
                    print(f"[Hợp Gu] Đã nạp {len(products)} sản phẩm từ: {path}")
                    return products
            except Exception as exc:
                print(f"[Cảnh báo] Không thể đọc {path}: {exc}")

    # Fallback to export script if parquet exists
    parquet_candidates = [
        base_dir / "data" / "processed" / "products.parquet",
        base_dir / "hopgu_v2" / "data" / "processed" / "products.parquet",
    ]
    for pq_path in parquet_candidates:
        if pq_path.is_file():
            try:
                import pandas as pd
                df = pd.read_parquet(pq_path)
                catalog = []
                for _, row in df.iterrows():
                    pid = str(row.get("product_id", "")).strip()
                    catalog.append({
                        "product_id": pid,
                        "product_name": str(row.get("product_name", "")),
                        "category": str(row.get("category", "")),
                        "price_vnd": int(row.get("price_vnd", 0)),
                        "rating": float(row.get("rating", 5.0)),
                        "review_count": int(row.get("review_count", 0)),
                        "sizes": ["S", "M", "L", "XL"] if "Thời trang" in str(row.get("category", "")) else ["Tiêu chuẩn"],
                        "material": f"Thương hiệu {row.get('brand_name')}" if pd.notna(row.get("brand_name")) else "Chính hãng Tiki",
                        "image_uri": f"/data/images/{pid}.jpg",
                        "image_url": str(row.get("image_url", "")),
                        "review_snippet": str(row.get("reviews_text", [""]))[0] if isinstance(row.get("reviews_text"), list) and row.get("reviews_text") else "",
                    })
                print(f"[Hợp Gu] Đã chuyển đổi trực tiếp {len(catalog)} sản phẩm từ {pq_path}")
                return catalog
            except Exception as err:
                print(f"[Cảnh báo] Không thể nạp Parquet: {err}")

    return []


def build_standalone_html(base_dir: Path, catalog: List[Dict[str, Any]]) -> str:
    """
    Tạo nội dung HTML đã nhúng sẵn toàn bộ dữ liệu sản phẩm để có thể gửi hoặc mở offline.
    """
    html_path = find_frontend_path(base_dir)
    html_text = html_path.read_text(encoding="utf-8")
    if catalog:
        injected_script = (
            "<script>\n"
            f"window.__HOPGU_INJECTED_CATALOG__ = {json.dumps(catalog, ensure_ascii=False)};\n"
            "</script>\n</head>"
        )
        if "</head>" in html_text:
            html_text = html_text.replace("</head>", injected_script, 1)

    output_path = base_dir / STANDALONE_OUTPUT_FILE
    output_path.write_text(html_text, encoding="utf-8")
    print(f"[Hợp Gu] Đã xuất bản đầy đủ độc lập tại: {output_path.resolve()}")
    return html_text


class HopGuRequestHandler(BaseHTTPRequestHandler):
    """Bộ xử lý HTTP phục vụ trang web, ảnh tĩnh và API cửa hàng Hợp Gu Atelier."""

    html_bundle: str = ""
    catalog_data: List[Dict[str, Any]] = []
    orders_log: List[Dict[str, Any]] = []
    base_dir: Path = Path(".")

    def _send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html_text: str, status: int = 200) -> None:
        body = html_text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path

        if route in ("/", "/index.html", f"/{FRONTEND_FILE}", f"/{ALT_FRONTEND_FILE}", f"/{STANDALONE_OUTPUT_FILE}"):
            self._send_html(self.html_bundle)
            return

        if route == "/api/catalog":
            self._send_json({"count": len(self.catalog_data), "products": self.catalog_data})
            return

        # Phục vụ ảnh local
        if route.startswith("/data/images/") or route.startswith("/images/"):
            filename = Path(route).name
            candidate_dirs = [
                self.base_dir / "data" / "images",
                self.base_dir / "hopgu_v2" / "data" / "images",
                self.base_dir.parent / "hopgu_v2" / "data" / "images",
            ]
            for c_dir in candidate_dirs:
                img_path = c_dir / filename
                if img_path.is_file():
                    ext = img_path.suffix.lower()
                    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else ("image/png" if ext == ".png" else "application/octet-stream")
                    content = img_path.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", mime)
                    self.send_header("Content-Length", str(len(content)))
                    self.send_header("Cache-Control", "public, max-age=86400")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(content)
                    return
            self._send_json({"error": "Không tìm thấy hình ảnh"}, status=404)
            return

        if route == "/api/search":
            params = parse_qs(parsed.query)
            q = (params.get("q", [""])[0] or "").lower().strip()
            q_norm = ascii_fold(q)
            cat = (params.get("category", [""])[0] or "").strip()
            max_price = int(params.get("max_price", ["0"])[0] or 0)

            results = []
            for item in self.catalog_data:
                item_cat = str(item.get("category", ""))
                item_cat_slug = str(item.get("category_slug", ""))
                if cat and (item_cat != cat and item_cat_slug != cat):
                    continue
                if max_price > 0 and int(item.get("price_vnd", 0)) > max_price:
                    continue
                if q:
                    pname = str(item.get("product_name", ""))
                    pname_lower = pname.lower()
                    pname_norm = ascii_fold(pname)
                    if q not in pname_lower and q_norm not in pname_norm:
                        continue
                results.append(item)

            self._send_json({"query": q, "count": len(results), "results": results})
            return

        self._send_json({"error": "Không tìm thấy đường dẫn"}, status=404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/checkout":
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            try:
                order = json.loads(raw_body)
            except Exception:
                order = {"raw": raw_body}

            self.orders_log.append(order)
            
            # Ghi nhận log đơn hàng ra file để đối soát
            try:
                log_file = self.base_dir / "data" / "orders_log.json"
                log_file.write_text(json.dumps(self.orders_log, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass

            self._send_json(
                {
                    "status": "success",
                    "message": "Đơn hàng của bạn đã được Hợp Gu Atelier tiếp nhận thành công!",
                    "order_index": len(self.orders_log),
                    "items_count": len(order.get("items", [])),
                }
            )
            return

        self._send_json({"error": "Phương thức không hỗ trợ"}, status=404)


def main() -> None:
    parser = argparse.ArgumentParser(description="Máy chủ & Đóng gói toàn bộ Hợp Gu Atelier")
    parser.add_argument("--host", default="127.0.0.1", help="Địa chỉ host (mặc định: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Cổng chạy web (mặc định: 8000)")
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Chỉ đóng gói ra file hopgu_standalone_full.html rồi thoát",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Tự động mở trình duyệt web khi khởi động máy chủ",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    catalog = load_local_catalog_if_exists(base_dir)
    bundled_html = build_standalone_html(base_dir, catalog)

    if args.build_only:
        return

    HopGuRequestHandler.base_dir = base_dir
    HopGuRequestHandler.html_bundle = bundled_html
    HopGuRequestHandler.catalog_data = catalog

    url = f"http://{args.host}:{args.port}"
    print("=" * 60)
    print("✦ HỢP GU ATELIER — MÁY CHỦ TRÌNH DIỄN & LOOKBOOK EDITORIAL ✦")
    print("=" * 60)
    print(f"✦ Máy chủ đang chạy tại: {url}")
    print(f"✦ Danh mục nạp: {len(catalog)} sản phẩm thật")
    print(f"✦ Bản độc lập: {STANDALONE_OUTPUT_FILE}")
    print("✦ Nhấn Ctrl+C để dừng máy chủ.")
    print("=" * 60)

    if args.open:
        webbrowser.open(url)

    server = ThreadingHTTPServer((args.host, args.port), HopGuRequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Hợp Gu] Đã dừng máy chủ thành công.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
