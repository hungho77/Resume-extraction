import os
import re
from typing import Dict, List, Any
from pathlib import Path
import logging

from docling.document_converter import DocumentConverter

from .config import docling_config
from .client import SmolDocLingClient, LLMClient, ResumeProcessor

logger = logging.getLogger(__name__)


class ResumeDocumentProcessor:
    """PDF document processor for resume parsing with markdown export and entity extraction"""

    def __init__(self):
        self.converter = DocumentConverter()
        self.use_ocr = docling_config.use_ocr
        self.ocr_language = docling_config.ocr_language

        # Initialize specialized clients
        self.smoldocling_client = SmolDocLingClient()
        self.llm_client = LLMClient()
        self.resume_processor = ResumeProcessor()

        logger.info("ResumeDocumentProcessor initialized with specialized clients")

    def extract_text_from_pdf(self, file_path: str) -> tuple[str, bool]:
        """Extract text from PDF using DocLing with SmolDocLing VLM OCR fallback"""
        text = ""
        used_smoldocling = False

        try:
            # Primary method: Use DocLing for PDF processing
            logger.info(f"Using DocLing for PDF extraction: {file_path}")

            # Convert document with DocLing
            conversion_result = self.converter.convert(file_path)

            # Try to export to markdown first (preserves structure better)
            if hasattr(conversion_result, "document") and hasattr(
                conversion_result.document, "export_to_markdown"
            ):
                try:
                    text = conversion_result.document.export_to_markdown()
                    logger.info("DocLing markdown export successful")
                    logger.debug(f"Markdown content preview: {text[:200]}...")
                except Exception as e:
                    logger.warning(
                        f"DocLing markdown export failed: {e}, falling back to text extraction"
                    )
                    text = ""

            # Fallback to text extraction if markdown export failed or not available
            if not text.strip():
                if hasattr(conversion_result, "text"):
                    text = conversion_result.text
                elif hasattr(conversion_result, "content"):
                    text = conversion_result.content
                elif hasattr(conversion_result, "pages"):
                    # Extract text from all pages
                    text = "\n".join(
                        [
                            page.text
                            for page in conversion_result.pages
                            if hasattr(page, "text")
                        ]
                    )
                elif isinstance(conversion_result, dict):
                    text = conversion_result.get("text", "")
                else:
                    # Fallback: try to get text from the document object
                    text = str(conversion_result)

            logger.info("DocLing PDF extraction successful")

        except Exception as e:
            logger.error(f"DocLing PDF extraction failed: {e}")
            text = ""

        # If DocLing failed or returned empty text, try SmolDocLing VLM OCR
        if not text.strip():
            logger.info("DocLing returned empty text, trying SmolDocLing VLM OCR...")
            text = self.smoldocling_client.extract_text_from_pdf(file_path)
            if text:
                logger.info("SmolDocLing VLM OCR extraction successful")
                used_smoldocling = True
            else:
                logger.error("All PDF extraction methods failed")

        return text.strip(), used_smoldocling

    def extract_text(self, file_path: str) -> tuple[str, bool]:
        """Extract text from PDF files using DocLing with markdown export"""
        # Convert to Path for suffix checking, but keep original string for processing
        path_obj = Path(file_path)
        file_path_str = str(file_path)  # Keep original string

        if path_obj.suffix.lower() == ".pdf":
            return self.extract_text_from_pdf(file_path_str)
        else:
            raise ValueError(f"Only PDF files are supported. Got: {path_obj.suffix}")

    def _is_markdown_format(self, text: str) -> bool:
        """Check if text contains markdown formatting"""
        markdown_patterns = [
            r"^#+\s",  # Headers
            r"\*\*.*?\*\*",  # Bold text
            r"\*.*?\*",  # Italic text
            r"^\s*[-*+]\s",  # Lists
            r"^\s*\d+\.\s",  # Numbered lists
            r"\[.*?\]\(.*?\)",  # Links
            r"`.*?`",  # Inline code
            r"```",  # Code blocks
        ]

        for pattern in markdown_patterns:
            if re.search(pattern, text, re.MULTILINE):
                return True
        return False

    def preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing for resume parsing"""
        # Remove extra whitespace but preserve structure
        text = re.sub(r"\n\s*\n", "\n\n", text)  # Normalize paragraph breaks
        text = re.sub(r" +", " ", text)  # Normalize spaces

        # Remove special characters but keep important ones
        text = re.sub(r"[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\@\#\$\%\&\+]", "", text)

        # Normalize line breaks
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        return text.strip()

    def extract_entities_from_resume(self, file_path: str) -> Dict[str, Any]:
        """
        Extract structured entities from resume using LLM

        Args:
            file_path: Path to resume file

        Returns:
            Dictionary containing extracted entities and validation results
        """
        try:
            logger.info(f"Extracting entities from resume: {file_path}")

            # Step 1: Extract text from document
            raw_text, used_smoldocling = self.extract_text(file_path)
            if not raw_text.strip():
                return {"error": "No text extracted from document"}

            # Step 2: Check if text is in markdown format
            is_markdown = self._is_markdown_format(raw_text)
            if is_markdown:
                logger.info("Detected markdown format in extracted text")
            else:
                logger.info("Text is in plain format")

            # Step 3: Preprocess text
            processed_text = self.preprocess_text(raw_text)

            # Step 4: Extract entities using LLM
            entities = self.llm_client.extract_entities(processed_text)

            # Step 5: Validate results
            validation = self._validate_extracted_entities(entities, processed_text)

            # Step 6: Prepare result
            result = {
                "entities": entities,
                "validation": validation,
                "metadata": {
                    "file_path": file_path,
                    "file_size": os.path.getsize(file_path),
                    "text_length": len(processed_text),
                    "processing_method": "docling_smoldocling_llm_extraction"
                    if used_smoldocling
                    else "docling_llm_extraction",
                    "is_markdown_format": is_markdown,
                    "ocr_enabled": self.use_ocr,
                    "ocr_language": self.ocr_language,
                    "smoldocling_available": self.smoldocling_client.health_check(),
                    "llm_available": self.llm_client.health_check(),
                    "extracted_text_preview": processed_text[:500] + "..."
                    if len(processed_text) > 500
                    else processed_text,
                },
            }

            logger.info(f"Successfully extracted entities from: {file_path}")
            return result

        except Exception as e:
            logger.error(f"Error extracting entities from {file_path}: {e}")
            return {"error": str(e)}

    def _validate_extracted_entities(
        self, entities: Dict[str, Any], original_text: str
    ) -> Dict[str, Any]:
        """
        Validate extracted entities against original text

        Args:
            entities: Extracted entities
            original_text: Original resume text

        Returns:
            Validation results
        """
        validation = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "completeness_score": 0.0,
            "field_scores": {},
        }

        # Required fields for evaluation
        required_fields = [
            "name",
            "email",
            "phone",
            "skills",
            "education",
            "experience",
            "certifications",
            "languages",
        ]

        # Check completeness
        present_fields = 0
        for field in required_fields:
            if field in entities and entities[field]:
                if isinstance(entities[field], list):
                    if len(entities[field]) > 0:
                        present_fields += 1
                        validation["field_scores"][field] = 1.0
                    else:
                        validation["field_scores"][field] = 0.0
                        validation["warnings"].append(f"Empty {field} list")
                else:
                    if entities[field].strip():
                        present_fields += 1
                        validation["field_scores"][field] = 1.0
                    else:
                        validation["field_scores"][field] = 0.0
                        validation["warnings"].append(f"Empty {field}")
            else:
                validation["field_scores"][field] = 0.0
                validation["warnings"].append(f"Missing {field}")

        validation["completeness_score"] = present_fields / len(required_fields)

        # Specific validations
        if validation["completeness_score"] < 0.5:
            validation["warnings"].append("Low completeness score")

        # Validate email format
        if entities.get("email"):
            if not self._is_valid_email(entities["email"]):
                validation["errors"].append("Invalid email format")
                validation["field_scores"]["email"] = 0.0

        # Validate phone format
        if entities.get("phone"):
            if not self._is_valid_phone(entities["phone"]):
                validation["warnings"].append("Phone number format may be invalid")
                validation["field_scores"]["phone"] = 0.5

        # Check for consistency with original text
        self._check_text_consistency(entities, original_text, validation)

        return validation

    def _is_valid_email(self, email: str) -> bool:
        """Validate email format"""
        import re

        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Z|a-z]{2,}$"
        return re.match(pattern, email) is not None

    def _is_valid_phone(self, phone: str) -> bool:
        """Validate phone number format"""
        digits = "".join(filter(str.isdigit, phone))
        return len(digits) >= 10

    def _check_text_consistency(
        self, entities: Dict[str, Any], original_text: str, validation: Dict[str, Any]
    ):
        """Check if extracted entities are consistent with original text"""
        text_lower = original_text.lower()

        # Check if name appears in text
        if entities.get("name"):
            name_lower = entities["name"].lower()
            if name_lower not in text_lower:
                validation["warnings"].append("Name not found in original text")

        # Check if email appears in text
        if entities.get("email"):
            email_lower = entities["email"].lower()
            if email_lower not in text_lower:
                validation["warnings"].append("Email not found in original text")

        # Check if phone appears in text
        if entities.get("phone"):
            phone_clean = "".join(filter(str.isdigit, entities["phone"]))
            text_digits = "".join(filter(str.isdigit, original_text))
            if phone_clean not in text_digits:
                validation["warnings"].append("Phone number not found in original text")

    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Main method to process a document with entity extraction and validation"""
        try:
            # Extract entities from resume
            result = self.extract_entities_from_resume(file_path)

            if "error" in result:
                return result

            # Add additional metadata
            result["metadata"]["processing_timestamp"] = str(
                Path(file_path).stat().st_mtime
            )
            result["metadata"]["file_extension"] = Path(file_path).suffix.lower()

            return result

        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            return {"error": str(e)}

    def batch_process_resumes(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Process multiple resumes in batch

        Args:
            file_paths: List of resume file paths

        Returns:
            Dictionary containing results for all resumes
        """
        results = {
            "processed_files": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "results": {},
            "summary": {
                "average_completeness": 0.0,
                "total_entities_found": 0,
                "validation_errors": 0,
                "validation_warnings": 0,
            },
        }

        total_completeness = 0.0
        total_entities = 0
        total_errors = 0
        total_warnings = 0

        for file_path in file_paths:
            try:
                # Validate that it's a PDF file
                if not file_path.lower().endswith(".pdf"):
                    logger.warning(f"Skipping non-PDF file: {file_path}")
                    results["results"][file_path] = {
                        "error": f"Not a PDF file: {file_path}"
                    }
                    results["failed_extractions"] += 1
                    continue

                logger.info(f"Processing PDF resume: {file_path}")
                result = self.process_document(file_path)

                results["results"][file_path] = result
                results["processed_files"] += 1

                if "error" not in result:
                    results["successful_extractions"] += 1

                    # Update summary statistics
                    validation = result.get("validation", {})
                    total_completeness += validation.get("completeness_score", 0.0)
                    total_errors += len(validation.get("errors", []))
                    total_warnings += len(validation.get("warnings", []))

                    # Count total entities
                    entities = result.get("entities", {})
                    for field, value in entities.items():
                        if isinstance(value, list) and len(value) > 0:
                            total_entities += len(value)
                        elif isinstance(value, str) and value.strip():
                            total_entities += 1
                else:
                    results["failed_extractions"] += 1

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                results["results"][file_path] = {"error": str(e)}
                results["failed_extractions"] += 1

        # Calculate summary statistics
        if results["successful_extractions"] > 0:
            results["summary"]["average_completeness"] = (
                total_completeness / results["successful_extractions"]
            )
            results["summary"]["total_entities_found"] = total_entities
            results["summary"]["validation_errors"] = total_errors
            results["summary"]["validation_warnings"] = total_warnings

        return results
