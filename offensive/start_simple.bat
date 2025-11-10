@echo off
REM AEGIS‑C Cold War Offensive Launcher (Simple Version)

echo ⚔️  AEGIS‑C Cold War Offensive Toolkit
echo 🎯 For authorized red-team testing only
echo ==================================================

REM Check if virtual environment exists
if not exist "..\venv_windows\Scripts\python.exe" (
    echo ❌ Virtual environment not found. Please run from main directory first.
    pause
    exit /b 1
)

echo ✅ Using existing dependencies from main platform
echo 🚀 Starting Cold War Offensive Dashboard...
echo 🌐 Dashboard will be available at: http://localhost:8502
echo 🛑 Close this window to stop the dashboard
echo.

..\venv_windows\Scripts\python.exe -m streamlit run simple_dashboard.py --server.port 8502 --server.address 0.0.0.0

pause