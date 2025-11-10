#!/bin/bash

# AEGIS‑C Platform Startup Script
# This script starts all services for the AEGIS‑C counter‑AI platform

set -e

echo "🛡️  Starting AEGIS‑C Counter‑AI Platform..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check if service is healthy
check_service() {
    local service_name=$1
    local port=$2
    local max_attempts=30
    local attempt=1
    
    echo -n "🔍 Checking $service_name health..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
            echo -e " ${GREEN}✓${NC}"
            return 0
        fi
        
        echo -n "."
        sleep 2
        ((attempt++))
    done
    
    echo -e " ${RED}✗${NC}"
    echo -e "${RED}Error: $service_name failed to start within 60 seconds${NC}"
    return 1
}

# Function to start service in background
start_service() {
    local service_name=$1
    local port=$2
    local command=$3
    
    echo -e "${BLUE}🚀 Starting $service_name on port $port...${NC}"
    
    # Start service in background
    eval "$command" > "logs/$service_name.log" 2>&1 &
    local pid=$!
    
    # Store PID for potential cleanup
    echo $pid > "logs/$service_name.pid"
    
    echo "📋 $service_name started with PID: $pid"
}

# Check if required ports are available
check_ports() {
    local ports=(8010 8011 8012 8013 8014 8501 5432 6379 7474 7687 9000 4222)
    
    echo "🔍 Checking port availability..."
    
    for port in "${ports[@]}"; do
        if lsof -i :$port > /dev/null 2>&1; then
            echo -e "${RED}Error: Port $port is already in use${NC}"
            echo "Please stop the service using port $port and try again"
            exit 1
        fi
    done
    
    echo -e "${GREEN}✓ All required ports are available${NC}"
}

# Create logs directory
mkdir -p logs

# Check prerequisites
echo "🔍 Checking prerequisites..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running${NC}"
    echo "Please start Docker and try again"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi

# Check ports
check_ports

# Start infrastructure services
echo -e "${BLUE}🏗️  Starting infrastructure services...${NC}"

# Start Docker services
echo "🐳 Starting Docker services (PostgreSQL, Redis, Neo4j, MinIO, NATS)..."
docker-compose up -d

# Wait for infrastructure to be ready
echo "⏳ Waiting for infrastructure services to be ready..."
sleep 10

# Check infrastructure health
echo "🔍 Checking infrastructure health..."
docker-compose ps

# Install Python dependencies
echo -e "${BLUE}📦 Installing Python dependencies...${NC}"

# Install dependencies for each service
services=("detector" "fingerprint" "honeynet" "admission" "provenance" "console")

for service in "${services[@]}"; do
    echo "📦 Installing dependencies for $service..."
    if [ -f "services/$service/requirements.txt" ]; then
        pip install -r "services/$service/requirements.txt" > /dev/null 2>&1
    fi
done

# Start application services
echo -e "${BLUE}🚀 Starting application services...${NC}"

# Start Detector service
start_service "detector" 8010 "uvicorn services.detector.main:app --reload --port 8010 --host 0.0.0.0"

# Start Fingerprinting service  
start_service "fingerprint" 8011 "uvicorn services.fingerprint.main:app --reload --port 8011 --host 0.0.0.0"

# Start Honeynet service
start_service "honeynet" 8012 "uvicorn services.honeynet.main:app --reload --port 8012 --host 0.0.0.0"

# Start Admission Control service
start_service "admission" 8013 "uvicorn services.admission.main:app --reload --port 8013 --host 0.0.0.0"

# Start Provenance service
start_service "provenance" 8014 "uvicorn services.provenance.main:app --reload --port 8014 --host 0.0.0.0"

# Wait for services to start
echo "⏳ Waiting for services to initialize..."
sleep 5

# Check service health
echo -e "${BLUE}🏥 Checking service health...${NC}"

check_service "detector" 8010
check_service "fingerprint" 8011  
check_service "honeynet" 8012
check_service "admission" 8013
check_service "provenance" 8014

# Start Console
echo -e "${BLUE}🖥️  Starting Console UI...${NC}"
start_service "console" 8501 "streamlit run services/console/app.py --server.port 8501 --server.address 0.0.0.0"

# Wait for console to start
echo "⏳ Waiting for console to initialize..."
sleep 10

check_service "console" 8501

# Display success message
echo -e "${GREEN}🎉 AEGIS‑C Platform started successfully!${NC}"
echo ""
echo -e "${BLUE}📊 Service URLs:${NC}"
echo "  🖥️  Console UI:        http://localhost:8501"
echo "  🔍 Detector Service:   http://localhost:8010"
echo "  🆔 Fingerprinting:     http://localhost:8011"
echo "  🍯 Honeynet:           http://localhost:8012"
echo "  🛡️  Admission Control: http://localhost:8013"
echo "  📋 Provenance:         http://localhost:8014"
echo ""
echo -e "${BLUE}🗄️  Infrastructure:${NC}"
echo "  🐘 PostgreSQL:         localhost:5432"
echo "  🔴 Redis:              localhost:6379"
echo "  🔵 Neo4j:              http://localhost:7474"
echo "  🪣 MinIO:              http://localhost:9000"
echo "  📡 NATS:               http://localhost:8222"
echo ""
echo -e "${YELLOW}📋 Logs are available in the ./logs/ directory${NC}"
echo -e "${YELLOW}🛑 To stop all services, run: ./stop.sh${NC}"
echo ""
echo -e "${GREEN}🛡️  AEGIS‑C is ready for operations!${NC}"