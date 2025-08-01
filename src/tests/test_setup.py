#!/usr/bin/env python3
"""
Test script to verify vLLM setup and check for import issues
"""

import sys
import subprocess
import os

# Set CUDA device order to avoid warnings with mixed GPU setups
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def test_vllm_import():
    """Test if vLLM can be imported"""
    print("Testing vLLM import...")
    try:
        import vllm

        print(f"✓ vLLM imported successfully (version: {vllm.__version__})")
        return True
    except ImportError as e:
        print(f"✗ Failed to import vLLM: {e}")
        return False


def test_vllm_cli():
    """Test if vLLM CLI commands work"""
    print("\nTesting vLLM CLI commands...")

    vllm_paths = [
        ["python", "-m", "vllm.entrypoints.openai.api_server"],
        ["python", "-m", "vllm.entrypoints.api_server"],
        ["python", "-m", "vllm.api_server"],
        ["vllm", "serve"],
    ]

    working_path = None
    for path in vllm_paths:
        try:
            print(f"Testing: {' '.join(path)}")
            result = subprocess.run(
                path + ["--help"], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                print(f"✓ Working CLI: {' '.join(path)}")
                working_path = path
                break
            else:
                print(f"✗ Failed: {' '.join(path)}")
        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            FileNotFoundError,
        ) as e:
            print(f"✗ Error with {' '.join(path)}: {e}")

    return working_path


def test_gpu_availability():
    """Test GPU availability"""
    print("\nTesting GPU availability...")
    try:
        import torch

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"✓ CUDA available with {gpu_count} GPU(s)")

            # Get unique GPU types
            gpu_types = set()
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                gpu_types.add(gpu_name)
                print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")

            if len(gpu_types) > 1:
                print(f"⚠️  Mixed GPU types detected: {', '.join(gpu_types)}")
                print(
                    "   CUDA_DEVICE_ORDER=PCI_BUS_ID is set to ensure consistent ordering"
                )

            return gpu_count
        else:
            print("✗ CUDA not available")
            return 0
    except ImportError:
        print("✗ PyTorch not available")
        return 0


def test_model_loading():
    """Test if we can load a small model"""
    print("\nTesting model loading...")
    try:
        from vllm import LLM

        # Try to load a small model for testing
        _ = LLM(
            model="Qwen/Qwen3-0.6B",  # Small model for testing
            trust_remote_code=True,
            tensor_parallel_size=os.getenv("LLM_TENSOR_PARALLEL_SIZE", "1"),
            max_model_len=1024,
            gpu_memory_utilization=0.7,
            dtype="float16",
        )
        print("✓ Model loaded successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False


def main():
    """Run all tests"""
    print("vLLM Setup Test")
    print("=" * 50)

    # Test imports
    vllm_ok = test_vllm_import()

    # Test CLI
    cli_path = test_vllm_cli()

    # Test GPU
    gpu_count = test_gpu_availability()

    # Test model loading (only if imports work)
    model_ok = False
    if vllm_ok:
        model_ok = test_model_loading()

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"vLLM Import: {'✓' if vllm_ok else '✗'}")
    print(f"vLLM CLI: {'✓' if cli_path else '✗'}")
    print(f"GPU Available: {'✓' if gpu_count > 0 else '✗'} ({gpu_count} GPUs)")
    print(f"Model Loading: {'✓' if model_ok else '✗'}")

    if cli_path:
        print(f"\nWorking CLI command: {' '.join(cli_path)}")

    if gpu_count > 0:
        print("\nRecommended GPU configuration:")
        if gpu_count >= 2:
            print(f"  Multi-GPU: --gpus {gpu_count}")
        else:
            print("  Single GPU: --gpus 1")

    # Exit with error if critical components fail
    if not vllm_ok:
        print("\n❌ Critical: vLLM import failed")
        sys.exit(1)

    if not cli_path:
        print("\n❌ Critical: No working vLLM CLI found")
        sys.exit(1)

    if gpu_count == 0:
        print("\n⚠️  Warning: No GPUs available")

    print("\n✅ Setup test completed")


if __name__ == "__main__":
    main()
