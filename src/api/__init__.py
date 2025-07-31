"""
API modules for the resume parser service.
"""

from .resume_extraction_server import app as resume_extraction_app
from .resume_extraction_server import process_resume_pipeline

# Import vLLM server if it exists
try:
    from .vllm_server import app as vllm_app
    VLLM_SERVER_AVAILABLE = True
except ImportError:
    vllm_app = None
    VLLM_SERVER_AVAILABLE = False

__all__ = [
    "resume_extraction_app", 
    "vllm_app", 
    "process_resume_pipeline",
    "VLLM_SERVER_AVAILABLE"
]
