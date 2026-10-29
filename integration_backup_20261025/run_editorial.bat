@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"

echo [HỢP GU] Đang khởi chạy máy chủ Lookbook Editorial & Mua sắm cao cấp...

where py >nul 2>nul
if not errorlevel 1 (
    py -3.14 serve_editorial.py --open %*
    if not errorlevel 1 exit /b 0
    py -3.11 serve_editorial.py --open %*
    if not errorlevel 1 exit /b 0
    py serve_editorial.py --open %*
    if not errorlevel 1 exit /b 0
)

where python >nul 2>nul
if not errorlevel 1 (
    python serve_editorial.py --open %*
    exit /b %errorlevel%
)

echo [LỖI] Không tìm thấy Python trên hệ thống để khởi chạy máy chủ.
pause
exit /b 1
