#!/bin/bash

# Resume Extraction Gradio Demo Runner
# Provides a web interface for the resume extraction API

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEMO_FILE="$PROJECT_ROOT/demo/gradio_demo.py"
PORT=7860

echo -e "${BLUE}🚀 Resume Extraction Gradio Demo${NC}"
echo "=================================================="

# Check if we're in the right directory
if [[ ! -f "$DEMO_FILE" ]]; then
    echo -e "${RED}❌ Demo file not found: $DEMO_FILE${NC}"
    echo "Please run this script from the project root directory."
    exit 1
fi

# Check if gradio is installed
if ! python3 -c "import gradio" 2>/dev/null; then
    echo -e "${YELLOW}⚠️ Gradio not found. Installing...${NC}"
    pip install gradio
fi

# Check if requests is installed
if ! python3 -c "import requests" 2>/dev/null; then
    echo -e "${YELLOW}⚠️ Requests not found. Installing...${NC}"
    pip install requests
fi

# Check if API server is running (optional check)
echo -e "${BLUE}🔍 Checking API server status...${NC}"
if curl -s http://localhost:8001/health >/dev/null 2>&1; then
    echo -e "${GREEN}✅ API server is running on http://localhost:8001${NC}"
else
    echo -e "${YELLOW}⚠️ API server not detected on http://localhost:8001${NC}"
    echo -e "${YELLOW}   Make sure to start the API server first:${NC}"
    echo -e "${YELLOW}   bash src/scripts/start_resume_server.sh${NC}"
    echo ""
fi

# Function to cleanup on exit
cleanup() {
    echo -e "\n${BLUE}🛑 Stopping Gradio demo...${NC}"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

echo -e "${BLUE}📁 Project root: $PROJECT_ROOT${NC}"
echo -e "${BLUE}📄 Demo file: $DEMO_FILE${NC}"
echo -e "${BLUE}🌐 Port: $PORT${NC}"
echo ""

# Change to project root directory
cd "$PROJECT_ROOT"

echo -e "${GREEN}🚀 Starting Gradio demo...${NC}"
echo -e "${GREEN}📱 Web interface will be available at: http://localhost:$PORT${NC}"
echo -e "${GREEN}📱 Public URL (if available): http://0.0.0.0:$PORT${NC}"
echo ""
echo -e "${YELLOW}💡 Tips:${NC}"
echo -e "${YELLOW}   • Upload a PDF resume to test the extraction${NC}"
echo -e "${YELLOW}   • Use the API configuration to connect to your server${NC}"
echo -e "${YELLOW}   • Enable OCR for scanned documents${NC}"
echo -e "${YELLOW}   • Download JSON results for further processing${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the demo${NC}"
echo ""

# Run the Gradio demo
python3 "$DEMO_FILE" 