#!/usr/bin/env python3
"""
Configuration settings for the Resume Parser
"""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class DocLingConfig:
    """DocLing configuration"""
    use_ocr: bool = True
    ocr_language: str = "en"
    ocr_confidence: float = 0.5

@dataclass
class VLLMConfig:
    """vLLM configuration"""
    host: str = "localhost"
    port: int = 8000
    model_name: str = "Qwen/Qwen3-8B"
    max_tokens: int = 1024
    temperature: float = 0.7

@dataclass
class OpenAIConfig:
    """OpenAI configuration"""
    api_key: Optional[str] = None
    model: str = "gpt-3.5-turbo"
    max_tokens: int = 1024
    temperature: float = 0.7

@dataclass
class OCRConfig:
    """OCR configuration"""
    use_ocr: bool = True
    ocr_language: str = "en"
    ocr_confidence: float = 0.5
    ocr_gpu: bool = True

@dataclass
class LLMConfig:
    """LLM provider configuration"""
    provider: str = "vllm"  # "vllm", "openai", "mock"
    vllm: VLLMConfig = None
    openai: OpenAIConfig = None
    
    def __post_init__(self):
        if self.vllm is None:
            self.vllm = VLLMConfig()
        if self.openai is None:
            self.openai = OpenAIConfig()

# Initialize configurations
docling_config = DocLingConfig()
vllm_config = VLLMConfig()
openai_config = OpenAIConfig()
ocr_config = OCRConfig()
llm_config = LLMConfig()

# Load from environment variables
def load_config_from_env():
    """Load configuration from environment variables"""
    global docling_config, vllm_config, openai_config, ocr_config, llm_config
    
    # DocLing config
    docling_config.use_ocr = os.getenv("DOCLING_USE_OCR", "true").lower() == "true"
    docling_config.ocr_language = os.getenv("DOCLING_OCR_LANGUAGE", "en")
    docling_config.ocr_confidence = float(os.getenv("DOCLING_OCR_CONFIDENCE", "0.5"))
    
    # vLLM config
    vllm_config.host = os.getenv("VLLM_HOST", "localhost")
    vllm_config.port = int(os.getenv("VLLM_PORT", "8000"))
    vllm_config.model_name = os.getenv("VLLM_MODEL", "Qwen/Qwen3-8B")
    vllm_config.max_tokens = int(os.getenv("VLLM_MAX_TOKENS", "1024"))
    vllm_config.temperature = float(os.getenv("VLLM_TEMPERATURE", "0.7"))
    
    # OpenAI config
    openai_config.api_key = os.getenv("OPENAI_API_KEY")
    openai_config.model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    openai_config.max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "1024"))
    openai_config.temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
    
    # OCR config
    ocr_config.use_ocr = os.getenv("OCR_USE_OCR", "true").lower() == "true"
    ocr_config.ocr_language = os.getenv("OCR_LANGUAGE", "en")
    ocr_config.ocr_confidence = float(os.getenv("OCR_CONFIDENCE", "0.5"))
    ocr_config.ocr_gpu = os.getenv("OCR_GPU", "true").lower() == "true"
    
    # LLM config
    llm_config.provider = os.getenv("LLM_PROVIDER", "vllm")
    llm_config.vllm = vllm_config
    llm_config.openai = openai_config

# Load configuration on import
load_config_from_env() 