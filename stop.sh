#!/bin/bash

# AEGIS‑C Platform Stop Script
# This script stops all services for the AEGIS‑C counter‑AI platform

set -e

echo "🛑 Stopping AEGIS‑C Counter‑AI Platform..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to stop service by PID
stop_service() {
    local service_name=$1
    local pid_file="logs/$service_name.pid"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" > /dev/null 2>&1; then
            echo -e "${YELLOW}🛑 Stopping $service_name (PID: $pid)...${NC}"
            kill "$pid"
            
            # Wait for service to stop
            local count=0
            while kill -0 "$pid" > /dev/null 2>&1 && [ $count -lt 10 ]; do
                sleep 1
                ((count++))
            done
            
            # Force kill if still running
            if kill -0 "$pid" > /dev/null 2>&1; then
                echo -e "${RED}⚡ Force killing $service_name...${NC}"
                kill -9 "$pid"
            fi
            
            echo -e "${GREEN}✓ $service_name stopped${NC}"
        else
            echo -e "${YELLOW}⚠️  $service_name was not running${NC}"
        fi
        
        rm -f "$pid_file"
    else
        echo -e "${YELLOW}⚠️  No PID file found for $service_name${NC}"
    fi
}

# Stop application services
echo -e "${BLUE}🛑 Stopping application services...${NC}"

services=("detector" "fingerprint" "honeynet" "admission" "provenance" "console")

for service in "${services[@]}"; do
    stop_service "$service"
done

# Kill any remaining processes on the ports
echo -e "${BLUE}🧹 Cleaning up remaining processes...${NC}"

ports=(8010 8011 8012 8013 8014 8501)

for port in "${ports[@]}"; do
    local pid=$(lsof -ti:$port 2>/dev/null || true)
    if [ ! -z "$pid" ]; then
        echo -e "${YELLOW}🛑 Killing process on port $port (PID: $pid)...${NC}"
        kill -9 "$pid" 2>/dev/null || true
    fi
done

# Stop infrastructure services
echo -e "${BLUE}🐳 Stopping Docker services...${NC}"
docker-compose down

# Clean up
echo -e "${BLUE}🧹 Cleaning up...${NC}"

# Remove PID files
rm -f logs/*.pid

# Optionally clean up logs (uncomment if desired)
# rm -f logs/*.log

echo -e "${GREEN}✅ AEGIS‑C Platform stopped successfully!${NC}"
echo ""
echo -e "${BLUE}📊 All services have been stopped:${NC}"
echo "  🔍 Detector Service:   stopped"
echo "  🆔 Fingerprinting:     stopped"
echo "  🍯 Honeynet:           stopped"
echo "  🛡️  Admission Control: stopped"
echo "  📋 Provenance:         stopped"
echo "  🖥️  Console UI:        stopped"
echo "  🐳 Docker Services:    stopped"
echo ""
echo -e "${GREEN}🛡️  AEGIS‑C is now offline${NC}"