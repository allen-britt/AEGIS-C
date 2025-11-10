@echo off
title AEGIS-C Unified Platform Launcher
color 0A

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                    🛡️ AEGIS-C UNIFIED LAUNCHER                ║
echo ║              Adaptive Counter-AI Intelligence Platform         ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo Starting AEGIS-C platform with unified web interface...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.8+ first.
    pause
    exit /b 1
)

REM Check if we're in the right directory
if not exist "services\brain\main.py" (
    echo ❌ AEGIS-C files not found. Please run this from the aegis-c directory.
    pause
    exit /b 1
)

REM Check dependencies
echo 🔍 Checking dependencies...
python -c "import fastapi, uvicorn, streamlit, requests" >nul 2>&1
if errorlevel 1 (
    echo ❌ Missing dependencies. Installing...
    pip install fastapi uvicorn streamlit requests plotly pandas
    if errorlevel 1 (
        echo ❌ Failed to install dependencies.
        pause
        exit /b 1
    )
)

REM Start the unified launcher
echo.
echo 🚀 Launching AEGIS-C Unified Platform...
echo    This will start ALL services and open your web browser
echo    with the complete AEGIS-C interface.
echo.
echo 🌐 Your platform will be available at: http://localhost:8500
echo.
echo Press Ctrl+C to stop all services when done.
echo.

python launch.py

echo.
echo 👋 AEGIS-C Platform stopped.
pause
