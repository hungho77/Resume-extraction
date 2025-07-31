#!/bin/bash

# Resume Parser - Server Startup Script
# This script starts both the vLLM server and the API server

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
VLLM_PORT=8000
API_PORT=8080
VLLM_HOST="0.0.0.0"
API_HOST="0.0.0.0"

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to check if port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
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
        print_status $YELLOW "Killing process on port $port (PID: $pid)"
        kill -9 $pid 2>/dev/null
        sleep 2
    fi
}

# Function to start vLLM server
start_vllm_server() {
    print_status $BLUE "🚀 Starting vLLM server..."
    
    if check_port $VLLM_PORT; then
        print_status $YELLOW "⚠️  Port $VLLM_PORT is already in use"
        read -p "Do you want to kill the existing process? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            kill_port $VLLM_PORT
        else
            print_status $RED "❌ Cannot start vLLM server. Port $VLLM_PORT is occupied."
            return 1
        fi
    fi
    
    # Start vLLM server in background
    print_status $BLUE "Starting vLLM server on $VLLM_HOST:$VLLM_PORT..."
    nohup ./scripts/start_server.sh --cli > vllm.log 2>&1 &
    VLLM_PID=$!
    
    # Wait for server to start
    print_status $BLUE "Waiting for vLLM server to start..."
    for i in {1..30}; do
        if curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
            print_status $GREEN "✅ vLLM server started successfully (PID: $VLLM_PID)"
            return 0
        fi
        sleep 2
        echo -n "."
    done
    
    print_status $RED "❌ vLLM server failed to start"
    return 1
}

# Function to start API server
start_api_server() {
    print_status $BLUE "🚀 Starting API server..."
    
    if check_port $API_PORT; then
        print_status $YELLOW "⚠️  Port $API_PORT is already in use"
        read -p "Do you want to kill the existing process? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            kill_port $API_PORT
        else
            print_status $RED "❌ Cannot start API server. Port $API_PORT is occupied."
            return 1
        fi
    fi
    
    # Start API server in background
    print_status $BLUE "Starting API server on $API_HOST:$API_PORT..."
    nohup python -m uvicorn src.api.server:app --host $API_HOST --port $API_PORT > api.log 2>&1 &
    API_PID=$!
    
    # Wait for server to start
    print_status $BLUE "Waiting for API server to start..."
    for i in {1..15}; do
        if curl -s "http://localhost:$API_PORT/health" > /dev/null 2>&1; then
            print_status $GREEN "✅ API server started successfully (PID: $API_PID)"
            return 0
        fi
        sleep 1
        echo -n "."
    done
    
    print_status $RED "❌ API server failed to start"
    return 1
}

# Function to check server status
check_servers() {
    print_status $BLUE "🔍 Checking server status..."
    
    # Check vLLM server
    if curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
        print_status $GREEN "✅ vLLM server is running on port $VLLM_PORT"
    else
        print_status $RED "❌ vLLM server is not running"
    fi
    
    # Check API server
    if curl -s "http://localhost:$API_PORT/health" > /dev/null 2>&1; then
        print_status $GREEN "✅ API server is running on port $API_PORT"
    else
        print_status $RED "❌ API server is not running"
    fi
}

# Function to stop servers
stop_servers() {
    print_status $YELLOW "🛑 Stopping servers..."
    
    # Stop vLLM server
    kill_port $VLLM_PORT
    print_status $GREEN "✅ vLLM server stopped"
    
    # Stop API server
    kill_port $API_PORT
    print_status $GREEN "✅ API server stopped"
}

# Function to show logs
show_logs() {
    print_status $BLUE "📋 Server Logs"
    echo "=================================================="
    
    if [ -f "vllm.log" ]; then
        print_status $BLUE "vLLM Server Log (last 10 lines):"
        tail -10 vllm.log
        echo
    fi
    
    if [ -f "api.log" ]; then
        print_status $BLUE "API Server Log (last 10 lines):"
        tail -10 api.log
        echo
    fi
}

# Function to test the API
test_api() {
    print_status $BLUE "🧪 Testing API endpoints..."
    
    # Test health endpoint
    print_status $BLUE "Testing health endpoint..."
    health_response=$(curl -s "http://localhost:$API_PORT/health")
    if [ $? -eq 0 ]; then
        print_status $GREEN "✅ Health check passed"
        echo "$health_response" | jq '.' 2>/dev/null || echo "$health_response"
    else
        print_status $RED "❌ Health check failed"
    fi
    
    # Test vLLM status
    print_status $BLUE "Testing vLLM status..."
    vllm_response=$(curl -s "http://localhost:$API_PORT/vllm/status")
    if [ $? -eq 0 ]; then
        print_status $GREEN "✅ vLLM status check passed"
        echo "$vllm_response" | jq '.' 2>/dev/null || echo "$vllm_response"
    else
        print_status $RED "❌ vLLM status check failed"
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start       Start both vLLM and API servers"
    echo "  stop        Stop both servers"
    echo "  restart     Restart both servers"
    echo "  status      Check server status"
    echo "  test        Test API endpoints"
    echo "  logs        Show server logs"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start    # Start both servers"
    echo "  $0 status   # Check if servers are running"
    echo "  $0 test     # Test API functionality"
}

# Main function
main() {
    case "${1:-start}" in
        start)
            print_status $GREEN "🚀 Starting Resume Parser Servers"
            echo "=================================================="
            
            if start_vllm_server && start_api_server; then
                print_status $GREEN "✅ All servers started successfully!"
                echo ""
                print_status $BLUE "📋 Server Information:"
                print_status $BLUE "   vLLM Server: http://localhost:$VLLM_PORT"
                print_status $BLUE "   API Server:  http://localhost:$API_PORT"
                echo ""
                print_status $BLUE "📝 Next steps:"
                print_status $BLUE "   - Run tests: $0 test"
                print_status $BLUE "   - Check logs: $0 logs"
                print_status $BLUE "   - Stop servers: $0 stop"
            else
                print_status $RED "❌ Failed to start servers"
                exit 1
            fi
            ;;
        stop)
            print_status $YELLOW "🛑 Stopping Resume Parser Servers"
            echo "=================================================="
            stop_servers
            ;;
        restart)
            print_status $BLUE "🔄 Restarting Resume Parser Servers"
            echo "=================================================="
            stop_servers
            sleep 3
            main start
            ;;
        status)
            print_status $BLUE "📊 Resume Parser Server Status"
            echo "=================================================="
            check_servers
            ;;
        test)
            print_status $BLUE "🧪 Testing Resume Parser API"
            echo "=================================================="
            test_api
            ;;
        logs)
            show_logs
            ;;
        help|--help|-h)
            show_usage
            ;;
        *)
            print_status $RED "❌ Unknown command: $1"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@" 