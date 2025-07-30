#!/usr/bin/env python3
"""
vLLM Server Script for Resume Parser
Starts a vLLM server to host Qwen3-8B model for document processing.
"""

import os
import sys
import argparse
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.api_server import create_app
from vllm.usage.usage_lib import UsageContext
import uvicorn

# If you have a config.py, import vllm_config from there, else set model_name directly below
try:
    from src.core.config import vllm_config
    MODEL_NAME = vllm_config.model_name or "Qwen/Qwen3-8B"
    MAX_MODEL_LEN = getattr(vllm_config, 'max_model_len', 32768)
    GPU_MEMORY_UTILIZATION = getattr(vllm_config, 'gpu_memory_utilization', 0.7)
    TENSOR_PARALLEL_SIZE = getattr(vllm_config, 'tensor_parallel_size', 1)
    TRUST_REMOTE_CODE = getattr(vllm_config, 'trust_remote_code', True)
    HOST = getattr(vllm_config, 'host', 'localhost')
    PORT = getattr(vllm_config, 'port', 8000)
except ImportError:
    MODEL_NAME = "Qwen/Qwen3-8B"
    MAX_MODEL_LEN = 32768
    GPU_MEMORY_UTILIZATION = 0.7
    TENSOR_PARALLEL_SIZE = 1
    TRUST_REMOTE_CODE = True
    HOST = "localhost"
    PORT = 8000

def create_vllm_engine():
    """Create vLLM engine with Qwen3-8B configuration"""
    engine_args = AsyncEngineArgs(
        model=MODEL_NAME,  # Use Qwen3-8B
        trust_remote_code=TRUST_REMOTE_CODE,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        enforce_eager=True,  # For better compatibility
        disable_log_stats=True,
    )
    return AsyncLLMEngine.from_engine_args(engine_args)

def start_vllm_server():
    """Start vLLM server with Qwen3-8B"""
    print(f"Starting vLLM server with model: {MODEL_NAME}")
    print(f"Server will be available at: http://{HOST}:{PORT}")
    print(f"Note: Qwen3-8B requires ~16GB VRAM for best performance.")
    
    # Create engine
    engine = create_vllm_engine()
    
    # Create FastAPI app
    app = create_app(
        engine=engine,
        served_model_names=[MODEL_NAME],
        usage_context=UsageContext.OPENAI_API_SERVER,
    )
    
    # Start server
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info"
    )

def test_model():
    """Test the Qwen3-8B model with a simple prompt"""
    print(f"Testing {MODEL_NAME} model...")
    
    try:
        # Initialize model
        llm = LLM(
            model=MODEL_NAME,
            trust_remote_code=TRUST_REMOTE_CODE,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        )
        
        # Test prompt for resume extraction
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=512,
            stop=["<|im_end|>", "\n\n", "###", "END"]
        )
        
        prompt = """<|im_start|>system
You are a helpful assistant that extracts information from resumes. Extract the name and email from this text.
<|im_end|>
<|im_start|>user
Extract the name and email from this text: John Doe, Software Engineer, john.doe@email.com
<|im_end|>
<|im_start|>assistant
"""
        
        outputs = llm.generate([prompt], sampling_params)
        
        print("Model test successful!")
        print(f"Prompt: {prompt}")
        print(f"Response: {outputs[0].outputs[0].text}")
        
    except Exception as e:
        print(f"Model test failed: {e}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="vLLM Server for Resume Parser with Qwen3-8B")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test the model instead of starting server"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help="Model name to use (default: Qwen/Qwen3-8B)"
    )
    parser.add_argument(
        "--host",
        type=str,
        help="Host to bind to (overrides config)"
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Port to bind to (overrides config)"
    )
    
    args = parser.parse_args()
    
    # Override config with command line arguments
    global MODEL_NAME, HOST, PORT
    if args.model:
        MODEL_NAME = args.model
    if args.host:
        HOST = args.host
    if args.port:
        PORT = args.port
    
    if args.test:
        success = test_model()
        sys.exit(0 if success else 1)
    else:
        start_vllm_server()

if __name__ == "__main__":
    main() 