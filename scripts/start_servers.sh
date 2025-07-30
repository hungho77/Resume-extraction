#!/bin/bash

# Resume Parser - Server Startup Script with Qwen3-8B
# This script starts both the vLLM server and the API server

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if required files exist
if [ ! -f "scripts/vllm_server.py" ]; then
    print_error "scripts/vllm_server.py not found"
    exit 1
fi

if [ ! -f "src/api/server.py" ]; then
    print_error "src/api/server.py not found"
    exit 1
fi

# Function to check if a port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        return 0
    else
        return 1
    fi
}

# Function to kill process on port
kill_port() {
    local port=$1
    local pid=$(lsof -ti:$port)
    if [ ! -z "$pid" ]; then
        print_warning "Killing process on port $port (PID: $pid)"
        kill -9 $pid 2>/dev/null || true
        sleep 2
    fi
}

# Parse command line arguments
VLLM_PORT=8000
API_PORT=8080
MODEL="Qwen/Qwen3-8B"
VLLM_ONLY=false
API_ONLY=false
BACKGROUND=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --vllm-port)
            VLLM_PORT="$2"
            shift 2
            ;;
        --api-port)
            API_PORT="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --vllm-only)
            VLLM_ONLY=true
            shift
            ;;
        --api-only)
            API_ONLY=true
            shift
            ;;
        --background)
            BACKGROUND=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --vllm-port PORT     vLLM server port (default: 8000)"
            echo "  --api-port PORT      API server port (default: 8080)"
            echo "  --model MODEL        Model to use with vLLM (default: Qwen/Qwen3-30B-A3B-Instruct-2507-FP8)"
            echo "  --vllm-only          Start only vLLM server"
            echo "  --api-only           Start only API server"
            echo "  --background         Run servers in background"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check and kill existing processes
print_status "Checking for existing processes..."

if check_port $VLLM_PORT; then
    print_warning "Port $VLLM_PORT is in use"
    kill_port $VLLM_PORT
fi

if check_port $API_PORT; then
    print_warning "Port $API_PORT is in use"
    kill_port $API_PORT
fi

# Start vLLM server
if [ "$API_ONLY" = false ]; then
    print_status "Starting vLLM server on port $VLLM_PORT with model: $MODEL"
    print_warning "Note: Qwen3-30B-A3B-Instruct-2507-FP8 requires significant GPU memory (24GB+ VRAM recommended)"
    print_warning "This is a large model with 30.5B parameters - ensure you have adequate GPU resources"
    
    if [ "$BACKGROUND" = true ]; then
        python3 scripts/vllm_server.py --model "$MODEL" --port $VLLM_PORT &
        VLLM_PID=$!
        print_success "vLLM server started in background (PID: $VLLM_PID)"
    else
        if [ "$VLLM_ONLY" = true ]; then
            print_status "Starting vLLM server only..."
            python3 scripts/vllm_server.py --model "$MODEL" --port $VLLM_PORT
        else
            # Start vLLM in background for combined mode
            python3 scripts/vllm_server.py --model "$MODEL" --port $VLLM_PORT &
            VLLM_PID=$!
            print_success "vLLM server started in background (PID: $VLLM_PID)"
            
            # Wait for vLLM to start
            print_status "Waiting for vLLM server to start (this may take 30-60 seconds for Qwen3-8B)..."
            sleep 60
            
            # Test vLLM connection with retries
            MAX_RETRIES=15
            RETRY_COUNT=0
            
            while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
                if python3 -c "
import requests
try:
    response = requests.get('http://localhost:$VLLM_PORT/health', timeout=15)
    if response.status_code == 200:
        print('vLLM server is ready')
        exit(0)
    else:
        print('vLLM server responding but not ready')
        exit(1)
except Exception as e:
    print(f'vLLM server not ready yet: {e}')
    exit(1)
"; then
                    print_success "vLLM server is ready"
                    break
                else
                    RETRY_COUNT=$((RETRY_COUNT + 1))
                    print_warning "vLLM server not ready yet (attempt $RETRY_COUNT/$MAX_RETRIES)"
                    sleep 15
                fi
            done
            
            if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
                print_warning "vLLM server may not be ready yet, continuing anyway..."
            fi
        fi
    fi
fi

# Start API server
if [ "$VLLM_ONLY" = false ]; then
    print_status "Starting API server on port $API_PORT"
    
    if [ "$BACKGROUND" = true ]; then
        python3 -m src.api.server &
        API_PID=$!
        print_success "API server started in background (PID: $API_PID)"
    else
        if [ "$API_ONLY" = true ]; then
            print_status "Starting API server only..."
            python3 -m src.api.server
        else
            # Start API server in foreground
            print_status "Starting API server..."
            python3 -m src.api.server
        fi
    fi
fi

# If running in background, show status
if [ "$BACKGROUND" = true ]; then
    echo ""
    print_status "Servers started in background:"
    if [ "$API_ONLY" = false ]; then
        echo "  vLLM Server: http://localhost:$VLLM_PORT (PID: $VLLM_PID)"
        echo "  Model: $MODEL"
    fi
    if [ "$VLLM_ONLY" = false ]; then
        echo "  API Server:  http://localhost:$API_PORT (PID: $API_PID)"
    fi
    echo ""
    print_status "To stop servers, use: pkill -f 'python3.*server.py'"
    echo ""
    print_status "To check server status:"
    echo "  vLLM: curl http://localhost:$VLLM_PORT/health"
    echo "  API:  curl http://localhost:$API_PORT/health"
    echo ""
    print_status "To test the system:"
    echo "  python tests/test_qwen_resume.py"
    echo "  python scripts/main.py --check-vllm"
fi

print_success "Setup complete!"
print_status "Remember: Qwen3-8B requires ~16GB VRAM for best performance. Monitor with 'nvidia-smi'" 