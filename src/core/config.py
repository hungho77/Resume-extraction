import os
import torch
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class VLLMConfig:
    """Configuration for vLLM server with Qwen3-30B-A3B-Instruct-2507-FP8"""
    model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"  # Updated to Qwen3-30B
    host: str = "localhost"
    port: int = 8000
    tensor_parallel_size: int = 1
    max_model_len: int = 32768  # Increased for Qwen3-30B
    gpu_memory_utilization: float = 0.7  # Adjusted for Qwen3-30B
    trust_remote_code: bool = True

@dataclass
class DocLingConfig:
    """Configuration for DocLing document processing with OCR"""
    max_length: int = 512
    batch_size: int = 8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_ocr: bool = True  # Enable OCR for PDF processing
    ocr_language: str = "eng"  # OCR language

@dataclass
class AppConfig:
    """Main application configuration"""
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = True
    
    # File upload settings
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_extensions: list = None
    
    # Processing settings
    chunk_size: int = 1000
    overlap_size: int = 200
    
    def __post_init__(self):
        if self.allowed_extensions is None:
            self.allowed_extensions = ['.pdf', '.docx', '.txt', '.doc']

# Global configuration instances
vllm_config = VLLMConfig()
docling_config = DocLingConfig()
app_config = AppConfig()

# Environment variables override
def load_env_config():
    """Load configuration from environment variables"""
    global vllm_config, docling_config, app_config
    
    # vLLM config
    vllm_config.model_name = os.getenv("VLLM_MODEL", vllm_config.model_name)
    vllm_config.host = os.getenv("VLLM_HOST", vllm_config.host)
    vllm_config.port = int(os.getenv("VLLM_PORT", vllm_config.port))
    
    # App config
    app_config.host = os.getenv("APP_HOST", app_config.host)
    app_config.port = int(os.getenv("APP_PORT", app_config.port))
    app_config.debug = os.getenv("APP_DEBUG", "true").lower() == "true"

# Load environment configuration
load_env_config() 