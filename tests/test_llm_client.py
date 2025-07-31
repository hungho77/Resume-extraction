#!/usr/bin/env python3
"""
Test script for the unified LLM client
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.core.client import LLMClient, ResumeLLMProcessor

def test_llm_client():
    """Test the unified LLM client"""
    print("🧪 Testing Unified LLM Client")
    print("=" * 50)
    
    # Test vLLM client
    print("\n📡 Testing vLLM Client:")
    try:
        vllm_client = LLMClient("vllm")
        if vllm_client.health_check():
            print("✅ vLLM client is healthy")
            
            # Test generation
            response = vllm_client.generate("Hello, how are you?", max_tokens=50)
            print(f"✅ vLLM generation successful: {response[:100]}...")
        else:
            print("❌ vLLM client is not available")
    except Exception as e:
        print(f"❌ vLLM client error: {e}")
    
    # Test OpenAI client (if API key is available)
    print("\n🤖 Testing OpenAI Client:")
    try:
        openai_client = LLMClient("openai")
        if openai_client.health_check():
            print("✅ OpenAI client is healthy")
            
            # Test generation
            response = openai_client.generate("Hello, how are you?", max_tokens=50)
            print(f"✅ OpenAI generation successful: {response[:100]}...")
        else:
            print("❌ OpenAI client is not available (check API key)")
    except Exception as e:
        print(f"❌ OpenAI client error: {e}")
    
    # Test ResumeLLMProcessor
    print("\n📄 Testing ResumeLLMProcessor:")
    try:
        processor = ResumeLLMProcessor("vllm")
        if processor.llm_client.health_check():
            print("✅ ResumeLLMProcessor is healthy")
            
            # Test specific info extraction
            sample_text = "John Doe is a software engineer with 5 years of experience in Python and JavaScript."
            skills = processor.extract_specific_info(sample_text, "skills")
            print(f"✅ Skills extraction: {skills}")
        else:
            print("❌ ResumeLLMProcessor is not available")
    except Exception as e:
        print(f"❌ ResumeLLMProcessor error: {e}")

if __name__ == "__main__":
    test_llm_client() 