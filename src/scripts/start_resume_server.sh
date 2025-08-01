#!/bin/bash

# Resume Extraction Server Starter
# =============================================================================

# Default configuration
DEFAULT_PORT=8001
DEFAULT_HOST="0.0.0.0"

# Parse command line arguments
PORT=${1:-$DEFAULT_PORT}
HOST=${2:-$DEFAULT_HOST}

echo "🚀 Starting Resume Extraction Server"
echo "=================================================="
echo "🌐 Host: $HOST"
echo "🔌 Port: $PORT"
echo ""

# Check if we're in the right directory
if [ ! -f "requirements.txt" ] && [ ! -f "../requirements.txt" ]; then
    echo "❌ Error: Please run this script from the Resume-extraction directory"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python3 is not installed or not in PATH"
    exit 1
fi

# Check if required files exist
SERVER_FILE="src/api/resume_extraction_server.py"
if [ ! -f "$SERVER_FILE" ]; then
    echo "❌ Error: Server file not found: $SERVER_FILE"
    exit 1
fi

echo "✅ Server file found: $SERVER_FILE"

# Create output directory if it doesn't exist
OUTPUT_DIR="output"
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "📁 Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

echo "✅ Output directory: $OUTPUT_DIR"

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: Virtual environment not detected"
    echo "   Consider activating your virtual environment first"
fi

# Check if dependencies are installed
echo "🔍 Checking dependencies..."
if ! python3 -c "import fastapi, uvicorn" 2>/dev/null; then
    echo "❌ Error: Required dependencies not installed"
    echo "   Please run: pip install -r requirements.txt"
    exit 1
fi

echo "✅ Dependencies check passed"

# Check if port is available
if command -v netstat &> /dev/null; then
    if netstat -tuln | grep -q ":$PORT "; then
        echo "⚠️  Warning: Port $PORT is already in use"
        echo "   You can specify a different port: ./scripts/start_resume_server.sh 8001"
    fi
fi

# Test server import
echo "🔍 Testing server import..."
if ! python3 -c "from src.api.resume_extraction_server import app; print('✅ Server import successful')" 2>/dev/null; then
    echo "❌ Error: Failed to import server module"
    exit 1
fi

echo "✅ Server import successful"

# Start the server
echo ""
echo "🌐 Starting server on http://$HOST:$PORT"
echo "📚 API Documentation: http://$HOST:$PORT/docs"
echo "🔍 Health Check: http://$HOST:$PORT/health"
echo "📊 Status: http://$HOST:$PORT/status"
echo ""
echo "Available endpoints:"
echo "  POST /extract              - Extract resume information"
echo "  POST /extract/batch        - Extract from multiple resumes"
echo "  POST /extract/specific     - Extract specific field"
echo "  GET  /health              - Health check"
echo "  GET  /status              - System status"
echo ""
echo "Usage examples:"
echo "  curl -X POST http://$HOST:$PORT/extract -F 'file=@resume.pdf'"
echo "  curl -X GET http://$HOST:$PORT/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server
python3 -m uvicorn src.api.resume_extraction_server:app \
    --host "$HOST" \
    --port "$PORT" \
    --log-level info 