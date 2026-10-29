@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"

if exist "hopgu_v2\run_editorial.bat" (
    cd hopgu_v2
    call run_editorial.bat %*
    exit /b %errorlevel%
)

if exist "serve_editorial.py" (
    py -3.14 serve_editorial.py --open %*
    exit /b %errorlevel%
)
