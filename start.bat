@echo off

REM AEGIS‑C Platform Startup Script (Windows)
REM This script starts all services for the AEGIS‑C counter‑AI platform

echo 🛡️  Starting AEGIS‑C Counter‑AI Platform...

REM Create logs directory
if not exist logs mkdir logs

REM Check prerequisites
echo 🔍 Checking prerequisites...

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo Error: Docker is not running
    echo Please start Docker Desktop and try again
    pause
    exit /b 1
)

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Start infrastructure services
echo 🏗️  Starting infrastructure services...

REM Start Docker services
echo 🐳 Starting Docker services ^(PostgreSQL, Redis, Neo4j, MinIO, NATS^)...
docker-compose up -d

REM Wait for infrastructure to be ready
echo ⏳ Waiting for infrastructure services to be ready...
timeout /t 10 /nobreak >nul

REM Check infrastructure health
echo 🔍 Checking infrastructure health...
docker-compose ps

REM Install Python dependencies
echo 📦 Installing Python dependencies...

REM Install dependencies for each service
echo 📦 Installing dependencies for detector...
pip install -r services\detector\requirements.txt >nul 2>&1

echo 📦 Installing dependencies for fingerprinting...
pip install -r services\fingerprint\requirements.txt >nul 2>&1

echo 📦 Installing dependencies for honeynet...
pip install -r services\honeynet\requirements.txt >nul 2>&1

echo 📦 Installing dependencies for admission...
pip install -r services\admission\requirements.txt >nul 2>&1

echo 📦 Installing dependencies for provenance...
pip install -r services\provenance\requirements.txt >nul 2>&1

echo 📦 Installing dependencies for console...
pip install -r services\console\requirements.txt >nul 2>&1

REM Start application services
echo 🚀 Starting application services...

REM Start Detector service
echo 🚀 Starting detector on port 8010...
start /B cmd /c "uvicorn services.detector.main:app --reload --port 8010 --host 0.0.0.0 > logs\detector.log 2>&1"

REM Start Fingerprinting service
echo 🚀 Starting fingerprinting on port 8011...
start /B cmd /c "uvicorn services.fingerprint.main:app --reload --port 8011 --host 0.0.0.0 > logs\fingerprint.log 2>&1"

REM Start Honeynet service
echo 🚀 Starting honeynet on port 8012...
start /B cmd /c "uvicorn services.honeynet.main:app --reload --port 8012 --host 0.0.0.0 > logs\honeynet.log 2>&1"

REM Start Admission Control service
echo 🚀 Starting admission on port 8013...
start /B cmd /c "uvicorn services.admission.main:app --reload --port 8013 --host 0.0.0.0 > logs\admission.log 2>&1"

REM Start Provenance service
echo 🚀 Starting provenance on port 8014...
start /B cmd /c "uvicorn services.provenance.main:app --reload --port 8014 --host 0.0.0.0 > logs\provenance.log 2>&1"

REM Wait for services to start
echo ⏳ Waiting for services to initialize...
timeout /t 5 /nobreak >nul

REM Start Console
echo 🖥️  Starting Console UI...
start /B cmd /c "streamlit run services\console\app.py --server.port 8501 --server.address 0.0.0.0 > logs\console.log 2>&1"

REM Wait for console to start
echo ⏳ Waiting for console to initialize...
timeout /t 10 /nobreak >nul

REM Display success message
echo.
echo 🎉 AEGIS‑C Platform started successfully!
echo.
echo 📊 Service URLs:
echo   🖥️  Console UI:        http://localhost:8501
echo   🔍 Detector Service:   http://localhost:8010
echo   🆔 Fingerprinting:     http://localhost:8011
echo   🍯 Honeynet:           http://localhost:8012
echo   🛡️  Admission Control: http://localhost:8013
echo   📋 Provenance:         http://localhost:8014
echo.
echo 🗄️  Infrastructure:
echo   🐘 PostgreSQL:         localhost:5432
echo   🔴 Redis:              localhost:6379
echo   🔵 Neo4j:              http://localhost:7474
echo   🪣 MinIO:              http://localhost:9000
echo   📡 NATS:               http://localhost:8222
echo.
echo 📋 Logs are available in the .\logs\ directory
echo 🛑 To stop all services, run: stop.bat
echo.
echo 🛡️  AEGIS‑C is ready for operations!
echo.
pause