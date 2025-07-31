#!/usr/bin/env python3
"""
Configuration settings for the Resume Parser
"""

import os
from dataclasses import dataclass


@dataclass
class DocLingConfig:
    """DocLing configuration"""

    use_ocr: bool = True
    ocr_language: str = "en"
    ocr_confidence: float = 0.5


# Initialize configurations
docling_config = DocLingConfig()


# Load from environment variables
def load_config_from_env():
    """Load configuration from environment variables"""
    global docling_config

    # DocLing config
    docling_config.use_ocr = os.getenv("DOCLING_USE_OCR", "true").lower() == "true"
    docling_config.ocr_language = os.getenv("DOCLING_OCR_LANGUAGE", "en")
    docling_config.ocr_confidence = float(os.getenv("DOCLING_OCR_CONFIDENCE", "0.5"))


# Load configuration on import
load_config_from_env()
