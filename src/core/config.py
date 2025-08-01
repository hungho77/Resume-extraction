#!/usr/bin/env python3
"""
Configuration settings for the Resume Parser
"""

import os
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()

@dataclass
class DocLingConfig:
    """DocLing configuration"""
    use_ocr: bool = True


# Initialize configurations
docling_config = DocLingConfig()


# Load from environment variables
def load_config_from_env():
    """Load configuration from environment variables"""
    global docling_config
    # DocLing config
    docling_config.use_ocr = os.getenv("DOCLING_USE_OCR", "true").lower() == "true"


# Load configuration on import
load_config_from_env()
