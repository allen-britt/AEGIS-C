@echo off

REM AEGIS‑C Platform Stop Script (Windows)
REM This script stops all services for the AEGIS‑C counter‑AI platform

echo 🛑 Stopping AEGIS‑C Counter‑AI Platform...

REM Stop application services by killing processes on ports
echo 🛑 Stopping application services...

echo 🛑 Stopping detector service...
for /f "tokens=5" %%a in ('netstat -aon ^| find ":8010" ^| find "LISTENING"') do taskkill /f /pid %%a >nul 2>&1

echo 🛑 Stopping fingerprinting service...
for /f "tokens=5" %%a in ('netstat -aon ^| find ":8011" ^| find "LISTENING"') do taskkill /f /pid %%a >nul 2>&1

echo 🛑 Stopping honeynet service...
for /f "tokens=5" %%a in ('netstat -aon ^| find ":8012" ^| find "LISTENING"') do taskkill /f /pid %%a >nul 2>&1

echo 🛑 Stopping admission service...
for /f "tokens=5" %%a in ('netstat -aon ^| find ":8013" ^| find "LISTENING"') do taskkill /f /pid %%a >nul 2>&1

echo 🛑 Stopping provenance service...
for /f "tokens=5" %%a in ('netstat -aon ^| find ":8014" ^| find "LISTENING"') do taskkill /f /pid %%a >nul 2>&1

echo 🛑 Stopping console service...
for /f "tokens=5" %%a in ('netstat -aon ^| find ":8501" ^| find "LISTENING"') do taskkill /f /pid %%a >nul 2>&1

REM Stop infrastructure services
echo 🐳 Stopping Docker services...
docker-compose down

REM Clean up
echo 🧹 Cleaning up...
if exist logs\*.pid del /q logs\*.pid

echo.
echo ✅ AEGIS‑C Platform stopped successfully!
echo.
echo 📊 All services have been stopped:
echo   🔍 Detector Service:   stopped
echo   🆔 Fingerprinting:     stopped
echo   🍯 Honeynet:           stopped
echo   🛡️  Admission Control: stopped
echo   📋 Provenance:         stopped
echo   🖥️  Console UI:        stopped
echo   🐳 Docker Services:    stopped
echo.
echo 🛡️  AEGIS‑C is now offline
echo.
pause