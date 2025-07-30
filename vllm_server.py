#!/usr/bin/env python3
"""
vLLM Server Script for Resume Parser
Starts a vLLM server to host Qwen3-30B-A3B-Instruct-2507-FP8 model for document processing.
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

from config import vllm_config

def create_vllm_engine():
    """Create vLLM engine with Qwen3-30B-A3B-Instruct-2507-FP8 configuration"""
    engine_args = AsyncEngineArgs(
        model="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",  # Using Qwen3-30B model
        trust_remote_code=vllm_config.trust_remote_code,
        tensor_parallel_size=vllm_config.tensor_parallel_size,
        max_model_len=vllm_config.max_model_len,
        gpu_memory_utilization=vllm_config.gpu_memory_utilization,
        enforce_eager=True,  # For better compatibility
        disable_log_stats=True,
    )
    
    return AsyncLLMEngine.from_engine_args(engine_args)

def start_vllm_server():
    """Start vLLM server with Qwen3-30B-A3B-Instruct-2507-FP8"""
    print(f"Starting vLLM server with model: Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")
    print(f"Server will be available at: http://{vllm_config.host}:{vllm_config.port}")
    print(f"Note: This model requires significant GPU memory (24GB+ VRAM recommended)")
    
    # Create engine
    engine = create_vllm_engine()
    
    # Create FastAPI app
    app = create_app(
        engine=engine,
        served_model_names=["Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"],
        usage_context=UsageContext.OPENAI_API_SERVER,
    )
    
    # Start server
    uvicorn.run(
        app,
        host=vllm_config.host,
        port=vllm_config.port,
        log_level="info"
    )

def test_model():
    """Test the Qwen3-30B-A3B-Instruct-2507-FP8 model with a simple prompt"""
    print("Testing Qwen3-30B-A3B-Instruct-2507-FP8 model...")
    
    try:
        # Initialize model
        llm = LLM(
            model="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
            trust_remote_code=vllm_config.trust_remote_code,
            tensor_parallel_size=vllm_config.tensor_parallel_size,
            max_model_len=vllm_config.max_model_len,
            gpu_memory_utilization=vllm_config.gpu_memory_utilization,
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
    parser = argparse.ArgumentParser(description="vLLM Server for Resume Parser with Qwen3-30B-A3B-Instruct-2507-FP8")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test the model instead of starting server"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
        help="Model name to use (default: Qwen/Qwen3-30B-A3B-Instruct-2507-FP8)"
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
    if args.model:
        vllm_config.model_name = args.model
    if args.host:
        vllm_config.host = args.host
    if args.port:
        vllm_config.port = args.port
    
    if args.test:
        success = test_model()
        sys.exit(0 if success else 1)
    else:
        start_vllm_server()

if __name__ == "__main__":
    main() 