#!/bin/bash

# Unified vLLM Server Startup Script
# Supports both single and multi-GPU deployments with automatic fallback

set -e

# Function to print output (using direct printf calls)
print_info() {
    printf "[INFO] %s\n" "$1"
}

print_success() {
    printf "[SUCCESS] %s\n" "$1"
}

print_warning() {
    printf "[WARNING] %s\n" "$1"
}

print_error() {
    printf "[ERROR] %s\n" "$1"
}

# Default configuration
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen3-8B"}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-2}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.8}
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-8000}
GPU_DEVICES=${GPU_DEVICES:-""}  # Custom GPU device selection

# Set CUDA device order to avoid warnings with mixed GPU setups
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check GPU availability
check_gpu() {
    print_info "Checking GPU availability..."
    if command_exists nvidia-smi; then
        print_info "GPU Information:"
        nvidia-smi --list-gpus
        print_info "GPU Memory Usage:"
        nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used --format=csv,noheader,nounits
        
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        print_info "Found $GPU_COUNT GPU(s)"
        
        if [ "$GPU_COUNT" -lt "$TENSOR_PARALLEL_SIZE" ]; then
            print_warning "Requested $TENSOR_PARALLEL_SIZE GPUs but only $GPU_COUNT available"
            print_info "Falling back to $GPU_COUNT GPUs"
            TENSOR_PARALLEL_SIZE=$GPU_COUNT
        fi
        
        # Check for mixed GPU types
        GPU_TYPES=$(nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq | wc -l)
        if [ "$GPU_TYPES" -gt 1 ]; then
            print_warning "Detected mixed GPU types. CUDA_DEVICE_ORDER=PCI_BUS_ID is set to ensure consistent ordering."
            
            # Analyze GPU memory and suggest optimal selection
            print_info "GPU Memory Analysis:"
            nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits | while IFS=',' read -r index name memory; do
                print_info "  GPU $index: $name (${memory}MB)"
            done
            
            # Suggest optimal GPU selection based on memory
            print_info "Recommended GPU selection for maximum memory:"
            nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits | sort -t',' -k3 -nr | while IFS=',' read -r index name memory; do
                print_info "  Priority: GPU $index ($name with ${memory}MB)"
            done
        fi
        
        return 0
    else
        print_error "nvidia-smi not found. Please install NVIDIA drivers."
        return 1
    fi
}

# Function to check vLLM installation
check_vllm() {
    print_info "Checking vLLM installation..."
    if python3 -c "import vllm" 2>/dev/null; then
        print_success "vLLM is installed"
        return 0
    else
        print_error "vLLM is not installed. Please install it first:"
        echo "pip install vllm"
        return 1
    fi
}

# Function to find the correct vLLM CLI command
find_vllm_cli() {
    # First check if vllm command exists and works
    if command -v vllm >/dev/null 2>&1; then
        if vllm -v >/dev/null 2>&1; then
            echo "vllm serve"
            return 0
        fi
    fi
    
    # Try different possible vLLM CLI paths
    local vllm_paths=(
        "python3 -m vllm.entrypoints.openai.api_server"
        "python3 -m vllm.entrypoints.api_server"
        "python3 -m vllm.api_server"
    )
    
    for cmd in "${vllm_paths[@]}"; do
        if eval "$cmd --help" >/dev/null 2>&1; then
            echo "$cmd"
            return 0
        fi
    done
    
    return 1
}

# Function to start server using vLLM CLI directly
start_vllm_cli() {
    print_info "Starting vLLM server using CLI directly..."
    
    print_info "Finding vLLM CLI command..."
    local vllm_cmd=$(find_vllm_cli)
    if [ $? -ne 0 ]; then
        print_error "Failed to find vLLM CLI command"
        return 1
    fi
    
    if [[ "$vllm_cmd" == "vllm serve" ]]; then
        print_success "Found vLLM CLI: $vllm_cmd"
    else
        print_success "Found vLLM python script: $vllm_cmd"
    fi

    # Build vLLM CLI command with arguments
    local cmd="$vllm_cmd"
    
    if [[ "$vllm_cmd" == "vllm serve" ]]; then
        # vllm serve has different syntax - model is positional argument
        cmd="$cmd $MODEL_NAME"
        cmd="$cmd --tensor-parallel-size $TENSOR_PARALLEL_SIZE"
        cmd="$cmd --max-model-len $MAX_MODEL_LEN"
        cmd="$cmd --gpu-memory-utilization $GPU_MEMORY_UTILIZATION"
        cmd="$cmd --host $HOST"
        cmd="$cmd --port $PORT"
        cmd="$cmd --trust-remote-code"
        # Note: vllm serve doesn't support --enforce-eager and --disable-log-stats
    else
        # Standard vLLM CLI syntax
        cmd="$cmd --model $MODEL_NAME"
        cmd="$cmd --tensor-parallel-size $TENSOR_PARALLEL_SIZE"
        cmd="$cmd --max-model-len $MAX_MODEL_LEN"
        cmd="$cmd --gpu-memory-utilization $GPU_MEMORY_UTILIZATION"
        cmd="$cmd --host $HOST"
        cmd="$cmd --port $PORT"
        cmd="$cmd --trust-remote-code"
        cmd="$cmd --enforce-eager"
        cmd="$cmd --disable-log-stats"
    fi
    
    # Add GPU device selection if specified
    if [ -n "$GPU_DEVICES" ]; then
        # Set CUDA_VISIBLE_DEVICES environment variable instead of --gpu-devices
        export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
        print_info "Using custom GPU devices: $GPU_DEVICES (via CUDA_VISIBLE_DEVICES)"
    fi
    
    printf "[INFO] Running command: %s\n" "$cmd"
    
    if eval "$cmd"; then
        return 0
    else
        printf "[ERROR] vLLM CLI server failed\n"
        return 1
    fi
}

# Function to start server using Python script
start_python_server() {
    print_info "Starting vLLM server using Python script..."
    
    # Get the directory where this script is located
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    
    # Use unified server script for both single and multi-GPU
    python3 "$SCRIPT_DIR/vllm_server.py" \
        --model "$MODEL_NAME" \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --host "$HOST" \
        --port "$PORT" \
        --cli
}

# Function to test the model
test_model() {
    print_info "Testing model..."
    
    # Get the directory where this script is located
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    
    # Use unified server script for both single and multi-GPU
    python3 "$SCRIPT_DIR/vllm_server.py" \
        --model "$MODEL_NAME" \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --test
}

# Function to show usage
show_usage() {
    printf "Usage: %s [OPTIONS]\n" "$0"
    printf "\n"
    printf "Options:\n"
    printf "  --test                    Test the model instead of starting server\n"
    printf "  --check-gpu               Check GPU availability and exit\n"
    printf "  --cli                     Use vLLM CLI directly (default)\n"
    printf "  --python                  Use Python script wrapper\n"
    printf "  --gpus N                  Number of GPUs to use (default: %s)\n" "$TENSOR_PARALLEL_SIZE"
    printf "  --model MODEL             Model name (default: %s)\n" "$MODEL_NAME"
    printf "  --max-model-len N         Max model length (default: %s)\n" "$MAX_MODEL_LEN"
    printf "  --gpu-memory-utilization F GPU memory utilization (default: %s)\n" "$GPU_MEMORY_UTILIZATION"
    printf "  --host HOST               Host to bind to (default: %s)\n" "$HOST"
    printf "  --port PORT               Port to bind to (default: %s)\n" "$PORT"
    printf "  --gpu-devices DEVICES     Specific GPU devices to use (e.g., '1,0' for GPU 1 first, sets CUDA_VISIBLE_DEVICES)\n"
    printf "  --help                    Show this help message\n"
    printf "\n"

    printf "Environment variables:\n"
    printf "  MODEL_NAME                Model name\n"
    printf "  TENSOR_PARALLEL_SIZE     Number of GPUs\n"
    printf "  MAX_MODEL_LEN            Max model length\n"
    printf "  GPU_MEMORY_UTILIZATION   GPU memory utilization\n"
    printf "  HOST                     Host to bind to\n"
    printf "  PORT                     Port to bind to\n"
    printf "  GPU_DEVICES              Specific GPU devices (e.g., '1,0', sets CUDA_VISIBLE_DEVICES)\n"
    printf "\n"
    printf "Examples:\n"
    printf "  %s --test\n" "$0"
    printf "  %s --gpus 2 --cli\n" "$0"
    printf "  %s --gpus 1 --python\n" "$0"
    printf "  %s --model Qwen/Qwen3-8B --gpus 2\n" "$0"
    printf "  %s --gpu-devices '1,0'  # Use GPU 1 first, then GPU 0\n" "$0"
    printf "  MODEL_NAME=Qwen/Qwen3-8B TENSOR_PARALLEL_SIZE=2 %s\n" "$0"
}

# Parse command line arguments
USE_CLI=true
USE_PYTHON=false
TEST_MODE=false
CHECK_GPU=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            TEST_MODE=true
            shift
            ;;
        --check-gpu)
            CHECK_GPU=true
            shift
            ;;
        --cli)
            USE_CLI=true
            USE_PYTHON=false
            shift
            ;;
        --python)
            USE_PYTHON=true
            USE_CLI=false
            shift
            ;;
        --gpus)
            TENSOR_PARALLEL_SIZE="$2"
            shift 2
            ;;
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --gpu-memory-utilization)
            GPU_MEMORY_UTILIZATION="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --gpu-devices)
            GPU_DEVICES="$2"
            shift 2
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main execution
print_info "Unified vLLM Server Startup Script"
print_info "Configuration:"
print_info "  Model: $MODEL_NAME"
print_info "  GPUs: $TENSOR_PARALLEL_SIZE"
print_info "  Max Model Length: $MAX_MODEL_LEN"
print_info "  GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
print_info "  Host: $HOST"
print_info "  Port: $PORT"

# Check GPU availability
if ! check_gpu; then
    print_error "GPU check failed"
    exit 1
fi

# Check vLLM installation
if ! check_vllm; then
    print_error "vLLM check failed"
    exit 1
fi

# Execute based on mode
if [ "$CHECK_GPU" = true ]; then
    print_success "GPU check completed"
    exit 0
elif [ "$TEST_MODE" = true ]; then
    test_model
elif [ "$USE_PYTHON" = true ]; then
    start_python_server
else
    start_vllm_cli
fi 