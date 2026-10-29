@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
call run_project.bat --check-only
if errorlevel 1 exit /b 1
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONIOENCODING=utf-8"
if not defined HOPGU_OFFLINE set "HOPGU_OFFLINE=1"
if /I "%~1"=="--check-only" exit /b 0
".venv\Scripts\python.exe" -B serve_editorial.py --open %*
exit /b %errorlevel%
