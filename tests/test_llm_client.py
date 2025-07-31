#!/usr/bin/env python3
"""
Test Unified LLM Client
Tests the unified LLM client with different configurations
"""

import sys
import os
import json
import time

# Add parent directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.client import LLMClient


def test_llm_client():
    """Test the unified LLM client"""
    print("🔧 Testing Unified LLM Client")
    print("=" * 50)

    # Test vLLM client
    try:
        vllm_client = LLMClient(base_url="http://localhost:8000/v1")
        health = vllm_client.health_check()
        print(f"📋 vLLM client: {'✅ Available' if health else '❌ Not available'}")

        if health:
            # Test generation
            prompt = "Hello, how are you?"
            response = vllm_client.generate(prompt, max_tokens=50)
            print(f"📝 Generation test: {response[:100]}...")

            # Test specific extraction
            text = "John Doe is a software engineer with 5 years of experience in Python and JavaScript."
            skills = vllm_client.extract_specific_info(text, "skills")
            print(f"💻 Skills extraction: {skills}")

        return health
    except Exception as e:
        print(f"❌ vLLM client error: {e}")
        return False


def test_openai_client():
    """Test OpenAI client"""
    print("\n🤖 Testing OpenAI Client")
    print("=" * 50)

    try:
        openai_client = LLMClient(
            base_url="https://api.openai.com/v1", api_key=os.getenv("LLM_API_KEY")
        )
        health = openai_client.health_check()
        print(f"🔍 Health check: {'✅ Available' if health else '❌ Not available'}")

        if health:
            # Test generation
            prompt = "Hello, how are you?"
            response = openai_client.generate(prompt, max_tokens=50)
            print(f"📝 Generation test: {response[:100]}...")

            # Test specific extraction
            text = "John Doe is a software engineer with 5 years of experience in Python and JavaScript."
            skills = openai_client.extract_specific_info(text, "skills")
            print(f"💻 Skills extraction: {skills}")

        return health
    except Exception as e:
        print(f"❌ OpenAI client error: {e}")
        return False


def test_client_performance():
    """Test client performance"""
    print("\n⚡ Testing Client Performance")
    print("=" * 50)

    try:
        client = LLMClient(base_url="http://localhost:8000/v1")

        if not client.health_check():
            print("❌ Client not available for performance test")
            return False

        # Test response time
        prompt = "Hello, how are you?"
        start_time = time.time()

        for i in range(3):
            _ = client.generate(prompt, max_tokens=50)
            elapsed = time.time() - start_time
            print(f"   Run {i + 1}: {elapsed:.2f}s")
            start_time = time.time()

        print("✅ Performance test completed")
        return True

    except Exception as e:
        print(f"❌ Performance test error: {e}")
        return False


def test_environment_configuration():
    """Test environment configuration"""
    print("\n⚙️  Testing Environment Configuration")
    print("=" * 50)

    # Test environment variables
    env_vars = {
        "LLM_BASE_URL": os.getenv("LLM_BASE_URL"),
        "LLM_API_KEY": os.getenv("LLM_API_KEY"),
        "LLM_MODEL": os.getenv("LLM_MODEL"),
    }

    print("🔧 Environment variables:")
    for key, value in env_vars.items():
        status = "Set" if value else "Not set"
        print(f"   {key}: {status}")

    # Test client with environment variables
    try:
        client = LLMClient()  # Uses environment variables
        health = client.health_check()
        print(
            f"🔍 Environment client: {'✅ Available' if health else '❌ Not available'}"
        )
        return health
    except Exception as e:
        print(f"❌ Environment client error: {e}")
        return False


def save_test_results(results, filename="llm_client_test_results.json"):
    """Save test results to JSON file"""
    try:
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to: {filename}")
    except Exception as e:
        print(f"❌ Failed to save results: {e}")


def main():
    """Run all tests"""
    print("🧪 Unified LLM Client Test Suite")
    print("=" * 60)

    results = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "tests": {}}

    # Run tests
    tests = [
        ("llm_client", test_llm_client),
        ("openai_client", test_openai_client),
        ("performance", test_client_performance),
        ("environment_config", test_environment_configuration),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{'=' * 20} {test_name.upper()} {'=' * 20}")
        try:
            result = test_func()
            results["tests"][test_name] = {
                "status": "PASSED" if result else "FAILED",
                "result": result,
            }
            if result:
                passed += 1
        except Exception as e:
            results["tests"][test_name] = {"status": "ERROR", "error": str(e)}

    # Summary
    print(f"\n{'=' * 60}")
    print("📊 Test Summary")
    print(f"{'=' * 60}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {total - passed}")
    print(f"📊 Total: {total}")

    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed")

    # Save results
    save_test_results(results)

    return passed == total


if __name__ == "__main__":
    main()
