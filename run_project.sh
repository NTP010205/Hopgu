#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

export UV_PYTHON_INSTALL_DIR="$PWD/.python"
export UV_CACHE_DIR="$PWD/.cache/uv"

if [[ ! -x .venv/bin/python ]]; then
  echo "[SETUP] Chưa có .venv; đang tạo môi trường Python 3.11..."
  if command -v uv >/dev/null 2>&1; then
    uv python install 3.11
    uv venv --python 3.11 --managed-python .venv
  elif command -v python3.11 >/dev/null 2>&1; then
    python3.11 -m venv .venv
  else
    echo "[ERROR] Cần Python 3.11 hoặc uv: https://docs.astral.sh/uv/" >&2
    exit 1
  fi
fi

.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version'
if ! .venv/bin/python -c 'import streamlit,pandas,pyarrow,numpy,PIL,requests,torch,transformers,sentence_transformers' >/dev/null 2>&1; then
  echo "[SETUP] Đang cài dependency từ requirements.txt..."
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python .venv/bin/python -r requirements.txt
  else
    .venv/bin/python -m pip install -r requirements.txt
  fi
fi

.venv/bin/python -c "from pathlib import Path; required=['data/processed/products.parquet','data/processed/reviews.parquet','data/embeddings/text_embeddings.npy','data/embeddings/text_manifest.json','data/embeddings/image_embeddings.npy','data/embeddings/image_manifest.json']; missing=[p for p in required if not Path(p).is_file()]; assert not missing, 'Thiếu artifact: '+', '.join(missing)"
export HOPGU_OFFLINE="${HOPGU_OFFLINE:-1}"

if [[ "${1:-}" == "--check-only" ]]; then
  echo "[OK] Python 3.11, dependency và artifact bắt buộc đã sẵn sàng."
  exit 0
fi

echo "[RUN] Mở Hợp Gu V2 tại http://localhost:8501"
exec .venv/bin/python -m streamlit run app.py --server.headless true --browser.gatherUsageStats false

