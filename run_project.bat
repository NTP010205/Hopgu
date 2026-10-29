@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"

set "UV_PYTHON_INSTALL_DIR=%CD%\.python"
set "UV_CACHE_DIR=%CD%\.cache\uv"

if exist ".venv\Scripts\python.exe" goto verify_python
echo [SETUP] Creating Python 3.11 environment...
where uv >nul 2>nul
if not errorlevel 1 goto create_with_uv
py -3.11 -c "import sys; assert sys.version_info.major == 3 and sys.version_info.minor == 11" >nul 2>nul
if errorlevel 1 goto missing_python
py -3.11 -m venv .venv
if errorlevel 1 exit /b 1
goto verify_python

:create_with_uv
uv python install 3.11
if errorlevel 1 exit /b 1
uv venv --python 3.11 --managed-python .venv
if errorlevel 1 exit /b 1

:verify_python
".venv\Scripts\python.exe" -c "import sys; assert sys.version_info.major == 3 and sys.version_info.minor == 11" >nul
if errorlevel 1 goto bad_version

".venv\Scripts\python.exe" -c "import streamlit,pandas,pyarrow,numpy,PIL,requests,torch,transformers,sentence_transformers" >nul 2>nul
if not errorlevel 1 goto verify_artifacts
echo [SETUP] Installing requirements.txt...
where uv >nul 2>nul
if not errorlevel 1 goto install_with_uv
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
goto verify_artifacts

:install_with_uv
uv pip install --python ".venv\Scripts\python.exe" -r requirements.txt
if errorlevel 1 exit /b 1

:verify_artifacts
".venv\Scripts\python.exe" -c "from pathlib import Path; required=['data/processed/products.parquet','data/processed/reviews.parquet','data/embeddings/text_embeddings.npy','data/embeddings/text_manifest.json','data/embeddings/image_embeddings.npy','data/embeddings/image_manifest.json']; missing=[p for p in required if not Path(p).is_file()]; assert not missing, 'Missing artifacts: '+', '.join(missing)"
if errorlevel 1 goto missing_artifacts

if not defined HOPGU_OFFLINE set "HOPGU_OFFLINE=1"
if /I "%~1"=="--check-only" goto check_ok

echo [RUN] Hop Gu V2: http://localhost:8501
".venv\Scripts\python.exe" -m streamlit run app.py --server.headless true --browser.gatherUsageStats false
exit /b %errorlevel%

:check_ok
echo [OK] Python 3.11, dependencies, and required artifacts are ready.
exit /b 0

:missing_python
echo [ERROR] Python 3.11 or uv is required. See https://docs.astral.sh/uv/
exit /b 1

:bad_version
echo [ERROR] .venv is not Python 3.11. Remove only .venv and run again.
exit /b 1

:missing_artifacts
echo [ERROR] Data or embeddings are missing. See README.md.
exit /b 1

