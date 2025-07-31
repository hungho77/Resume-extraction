#!/usr/bin/env python3
"""
Unified vLLM Server Script for Resume Parser
Starts a vLLM server with support for both single and multi-GPU configurations.
Compatible with vLLM 0.10.0 and later versions.
"""

import os
import sys
import argparse
import subprocess
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

# Set CUDA device order to avoid warnings with mixed GPU setups
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Default configuration
DEFAULT_MODEL_NAME = "Qwen/Qwen3-8B"
DEFAULT_MAX_MODEL_LEN = 32768
DEFAULT_GPU_MEMORY_UTILIZATION = 0.8
DEFAULT_TENSOR_PARALLEL_SIZE = 1  # Default to single GPU
DEFAULT_TRUST_REMOTE_CODE = True
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000


def check_gpu_availability():
    """Check if CUDA GPUs are available"""
    try:
        import torch

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"Found {gpu_count} CUDA GPU(s)")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            return gpu_count
        else:
            print("No CUDA GPUs available")
            return 0
    except ImportError:
        print("PyTorch not available, cannot check GPU status")
        return 0


def create_vllm_engine(
    model_name, tensor_parallel_size, max_model_len, gpu_memory_utilization
):
    """Create vLLM engine with configuration"""
    engine_args = AsyncEngineArgs(
        model=model_name,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=True,
        disable_log_stats=True,
        # Multi-GPU specific settings
        max_num_batched_tokens=8192 if tensor_parallel_size > 1 else 4096,
        max_num_seqs=256 if tensor_parallel_size > 1 else 128,
    )
    return AsyncLLMEngine.from_engine_args(engine_args)


def start_vllm_server(
    model_name, tensor_parallel_size, max_model_len, gpu_memory_utilization, host, port
):
    """Start vLLM server with configuration"""
    print(f"Starting vLLM server with model: {model_name}")
    print(f"Tensor parallel size: {tensor_parallel_size} GPUs")
    print(f"GPU memory utilization: {gpu_memory_utilization}")
    print(f"Server will be available at: http://{host}:{port}")

    # For vLLM 0.10.0, use CLI approach instead of API
    print("Using vLLM CLI approach for version 0.10.0")
    print("Please use the --cli flag or run vllm serve directly")
    print(
        f"Example: vllm serve {model_name} --tensor-parallel-size {tensor_parallel_size} --host {host} --port {port}"
    )


def test_model(model_name, tensor_parallel_size, max_model_len, gpu_memory_utilization):
    """Test the model with configuration"""
    print(f"Testing {model_name} model with {tensor_parallel_size} GPUs...")

    try:
        # Initialize model
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )

        # Test prompt for resume extraction
        sampling_params = SamplingParams(
            temperature=0.7, max_tokens=512, stop=["<|im_end|>", "\n\n", "###", "END"]
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


def run_vllm_cli(
    model_name, tensor_parallel_size, max_model_len, gpu_memory_utilization, host, port
):
    """Run vLLM using the CLI interface"""
    print(f"Starting vLLM CLI server with model: {model_name}")
    print(f"Tensor parallel size: {tensor_parallel_size} GPUs")
    print(f"Server will be available at: http://{host}:{port}")

    # Try different vLLM CLI paths for compatibility
    vllm_paths = [
        ["python3", "-m", "vllm.entrypoints.openai.api_server"],
        ["python3", "-m", "vllm.entrypoints.api_server"],
        ["python3", "-m", "vllm.api_server"],
        ["vllm", "serve"],
    ]

    cmd = None
    for path in vllm_paths:
        try:
            # Test if this path works
            if path[0] == "vllm" and path[1] == "serve":
                # vllm serve doesn't have --help, check if command exists
                subprocess.run(["which", "vllm"], capture_output=True, check=True)
                cmd = path
                print(f"Using vLLM CLI: {' '.join(cmd)}")
                break
            else:
                test_cmd = path + ["--help"]
                subprocess.run(test_cmd, capture_output=True, check=True, timeout=5)
                cmd = path
                print(f"Using vLLM CLI: {' '.join(cmd)}")
                break
        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            FileNotFoundError,
        ):
            continue

    if cmd is None:
        print("Error: Could not find working vLLM CLI command")
        return False

    # Build vLLM CLI command with arguments
    if cmd[0] == "vllm" and cmd[1] == "serve":
        # vllm serve has different syntax - model is positional argument
        cmd.extend([model_name])
        cmd.extend(
            [
                "--tensor-parallel-size",
                str(tensor_parallel_size),
                "--max-model-len",
                str(max_model_len),
                "--gpu-memory-utilization",
                str(gpu_memory_utilization),
                "--host",
                host,
                "--port",
                str(port),
                "--trust-remote-code",
            ]
        )
    else:
        # Standard vLLM CLI syntax
        cmd.extend(
            [
                "--model",
                model_name,
                "--tensor-parallel-size",
                str(tensor_parallel_size),
                "--max-model-len",
                str(max_model_len),
                "--gpu-memory-utilization",
                str(gpu_memory_utilization),
                "--host",
                host,
                "--port",
                str(port),
                "--trust-remote-code",
                "--enforce-eager",
                "--disable-log-stats",
            ]
        )

    print(f"Running command: {' '.join(cmd)}")

    try:
        # Run the vLLM CLI server
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"vLLM CLI server failed: {e}")
        return False
    except KeyboardInterrupt:
        print("\nServer stopped by user")
        return True

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Unified vLLM Server for Resume Parser"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Model name to use (default: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=DEFAULT_TENSOR_PARALLEL_SIZE,
        help=f"Number of GPUs to use for tensor parallelism (default: {DEFAULT_TENSOR_PARALLEL_SIZE})",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=DEFAULT_MAX_MODEL_LEN,
        help=f"Maximum model length (default: {DEFAULT_MAX_MODEL_LEN})",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
        help=f"GPU memory utilization (default: {DEFAULT_GPU_MEMORY_UTILIZATION})",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_HOST,
        help=f"Host to bind to (default: {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port to bind to (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--test", action="store_true", help="Test the model instead of starting server"
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Use vLLM CLI instead of Python API (recommended for production)",
    )
    parser.add_argument(
        "--check-gpu", action="store_true", help="Check GPU availability and exit"
    )

    args = parser.parse_args()

    # Check GPU availability if requested
    if args.check_gpu:
        gpu_count = check_gpu_availability()
        if gpu_count < args.tensor_parallel_size:
            print(
                f"Warning: Requested {args.tensor_parallel_size} GPUs but only {gpu_count} available"
            )
        sys.exit(0)

    # Validate tensor parallel size
    if args.tensor_parallel_size < 1:
        print("Error: tensor-parallel-size must be at least 1")
        sys.exit(1)

    # Check GPU availability
    gpu_count = check_gpu_availability()
    if gpu_count < args.tensor_parallel_size:
        print(
            f"Error: Requested {args.tensor_parallel_size} GPUs but only {gpu_count} available"
        )
        sys.exit(1)

    if args.test:
        success = test_model(
            args.model,
            args.tensor_parallel_size,
            args.max_model_len,
            args.gpu_memory_utilization,
        )
        sys.exit(0 if success else 1)
    elif args.cli:
        success = run_vllm_cli(
            args.model,
            args.tensor_parallel_size,
            args.max_model_len,
            args.gpu_memory_utilization,
            args.host,
            args.port,
        )
        sys.exit(0 if success else 1)
    else:
        start_vllm_server(
            args.model,
            args.tensor_parallel_size,
            args.max_model_len,
            args.gpu_memory_utilization,
            args.host,
            args.port,
        )


if __name__ == "__main__":
    main()
