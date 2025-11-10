#!/bin/bash

# AEGIS-C Unified Platform Launcher
# One script to start everything

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print banner
echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    🛡️ AEGIS-C UNIFIED LAUNCHER                ║"
echo "║              Adaptive Counter-AI Intelligence Platform         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo ""
echo "Starting AEGIS-C platform with unified web interface..."
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found. Please install Python 3.8+ first.${NC}"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "services/brain/main.py" ]; then
    echo -e "${RED}❌ AEGIS-C files not found. Please run this from the aegis-c directory.${NC}"
    exit 1
fi

# Check dependencies
echo -e "${YELLOW}🔍 Checking dependencies...${NC}"
if ! python3 -c "import fastapi, uvicorn, streamlit, requests" &> /dev/null; then
    echo -e "${YELLOW}❌ Missing dependencies. Installing...${NC}"
    pip3 install fastapi uvicorn streamlit requests plotly pandas
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Failed to install dependencies.${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}✅ Dependencies OK${NC}"

# Make script executable
chmod +x launch.py

# Start the unified launcher
echo ""
echo -e "${GREEN}🚀 Launching AEGIS-C Unified Platform...${NC}"
echo "   This will start ALL services and open your web browser"
echo "   with the complete AEGIS-C interface."
echo ""
echo -e "${BLUE}🌐 Your platform will be available at: http://localhost:8500${NC}"
echo ""
echo -e "${YELLOW}💡 Press Ctrl+C to stop all services when done.${NC}"
echo ""

python3 launch.py

echo ""
echo -e "${GREEN}👋 AEGIS-C Platform stopped.${NC}"
