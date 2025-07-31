#!/usr/bin/env python3
"""
Example usage of the unified LLM client
Shows how to use the same client for both vLLM and OpenAI
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.client import LLMClient, ResumeLLMProcessor

def example_vllm_usage():
    """Example using vLLM server"""
    print("🚀 Example: Using vLLM Server")
    print("=" * 40)
    
    # For vLLM, just set the base URL (no API key needed)
    client = LLMClient(
        base_url="http://localhost:8000",  # Your vLLM server
        model="Qwen/Qwen3-8B"
    )
    
    if client.health_check():
        print("✅ vLLM client is healthy")
        
        # Test generation
        response = client.generate("Hello, how are you?", max_tokens=50)
        print(f"✅ Response: {response}")
    else:
        print("❌ vLLM server not available")
        print("💡 Start vLLM server with: python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-8B --host 0.0.0.0 --port 8000")

def example_openai_usage():
    """Example using OpenAI API"""
    print("\n🤖 Example: Using OpenAI API")
    print("=" * 40)
    
    # Check if OpenAI API key is available
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        print("💡 Set your OpenAI API key in the .env file")
        return
    
    # For OpenAI, set both base URL and API key
    client = LLMClient(
        base_url="https://api.openai.com/v1",  # OpenAI API
        api_key=openai_api_key,    # Load from environment
        model="gpt-4o-mini"
    )
    
    if client.health_check():
        print("✅ OpenAI client is healthy")
        
        # Test generation
        response = client.generate("Hello, how are you?", max_tokens=50)
        print(f"✅ Response: {response}")
    else:
        print("❌ OpenAI client not available")
        print("💡 Check your OpenAI API key and network connection")

def example_resume_processing():
    """Example using ResumeLLMProcessor"""
    print("\n📄 Example: Resume Processing")
    print("=" * 40)
    
    # Create processor with vLLM
    processor = ResumeLLMProcessor(
        base_url="http://localhost:8000",
        model="Qwen/Qwen3-8B"
    )
    
    if processor.llm_client.health_check():
        print("✅ ResumeLLMProcessor is healthy")
        
        # Test specific info extraction
        sample_text = "John Doe is a software engineer with 5 years of experience in Python and JavaScript."
        skills = processor.extract_specific_info(sample_text, "skills")
        print(f"✅ Skills extraction: {skills}")
    else:
        print("❌ ResumeLLMProcessor is not available")
        print("💡 Make sure your LLM service is running")

def example_environment_config():
    """Example using environment variables"""
    print("\n⚙️  Example: Environment Configuration")
    print("=" * 40)
    
    # This will use settings from .env file
    client = LLMClient()  # No parameters - uses environment variables
    
    print(f"Base URL: {client.base_url}")
    print(f"Model: {client.model}")
    print(f"API Key: {'Set' if client.api_key else 'Not set'}")
    
    # Check specific environment variables
    openai_key = os.getenv("OPENAI_API_KEY")
    print(f"OPENAI_API_KEY: {'Set' if openai_key else 'Not set'}")
    
    if client.health_check():
        print("✅ Client is healthy")
    else:
        print("❌ Client is not available")

if __name__ == "__main__":
    print("🧪 Unified LLM Client Examples")
    print("=" * 50)
    
    example_vllm_usage()
    example_openai_usage()
    example_resume_processing()
    example_environment_config()
    
    print("\n💡 Key Points:")
    print("- Same client works for both vLLM and OpenAI")
    print("- Just change the base_url and api_key parameters")
    print("- For vLLM: only base_url needed")
    print("- For OpenAI: both base_url and api_key needed")
    print("- Environment variables can be used for configuration") 