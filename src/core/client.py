#!/usr/bin/env python3
"""
Enhanced LLM and SmolDocLing Clients for Resume Processing
Handles multiple formats, entity extraction, and result validation
"""

import os
import json
import logging
from typing import Dict, Any
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI imports
try:
    import openai
    from openai.types import Model

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# SmolDocLing imports
try:
    import torch
    from transformers import AutoProcessor, AutoModelForVision2Seq
    from transformers.image_utils import load_image
    from docling_core.types.doc import DoclingDocument
    from docling_core.types.doc.document import DocTagsDocument
    import fitz  # PyMuPDF for PDF image extraction

    SMOLDOCLING_AVAILABLE = True
except ImportError:
    SMOLDOCLING_AVAILABLE = False

logger = logging.getLogger(__name__)


class SmolDocLingClient:
    """SmolDocLing client for OCR processing of scanned PDFs"""

    def __init__(self, model_name: str = "ds4sd/SmolDocling-256M-preview"):
        self.model_name = model_name
        self.processor = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if SMOLDOCLING_AVAILABLE:
            try:
                logger.info(f"Initializing SmolDocLing client on {self.device}")
                self.processor = AutoProcessor.from_pretrained(model_name)
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    _attn_implementation="eager",
                ).to(self.device)
                logger.info("SmolDocLing client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize SmolDocLing client: {e}")
                self.processor = None
                self.model = None
        else:
            logger.warning("SmolDocLing packages not available")

    def health_check(self) -> bool:
        """Check if SmolDocLing is available and ready"""
        return self.processor is not None and self.model is not None

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using SmolDocLing OCR"""
        if not self.health_check():
            logger.warning("SmolDocLing not available for OCR processing")
            return ""

        try:
            logger.info(f"Processing PDF with SmolDocLing OCR: {pdf_path}")

            # Open PDF with PyMuPDF
            doc = fitz.open(pdf_path)
            all_text = ""

            for page_num in range(len(doc)):
                logger.info(f"Processing page {page_num + 1}/{len(doc)}")

                page = doc.load_page(page_num)

                # Convert page to image with higher resolution
                mat = fitz.Matrix(2.0, 2.0)  # Higher resolution for better OCR
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")

                # Save temporary image
                temp_image_path = f"/tmp/smoldocling_page_{page_num}.png"
                with open(temp_image_path, "wb") as f:
                    f.write(img_data)

                try:
                    # Load image
                    image = load_image(temp_image_path)

                    # Create input messages for SmolDocLing
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {
                                    "type": "text",
                                    "text": "Convert this page to docling format with all text content.",
                                },
                            ],
                        },
                    ]

                    # Prepare inputs
                    prompt = self.processor.apply_chat_template(
                        messages, add_generation_prompt=True
                    )
                    inputs = self.processor(
                        text=prompt, images=[image], return_tensors="pt"
                    )

                    # Move inputs to same device as model
                    device = next(self.model.parameters()).device
                    inputs = inputs.to(device)

                    # Generate outputs
                    with torch.no_grad():
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=8192,
                            do_sample=False,
                            temperature=0.0,
                        )

                    prompt_length = inputs.input_ids.shape[1]
                    trimmed_generated_ids = generated_ids[:, prompt_length:]
                    doctags = self.processor.batch_decode(
                        trimmed_generated_ids,
                        skip_special_tokens=False,
                    )[0].lstrip()

                    # Convert to DoclingDocument and extract text
                    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs(
                        [doctags], [image]
                    )
                    doc_ling = DoclingDocument.load_from_doctags(
                        doctags_doc, document_name=f"Page_{page_num + 1}"
                    )

                    page_text = doc_ling.export_to_markdown()
                    all_text += page_text + "\n\n"

                except Exception as e:
                    logger.error(f"Error processing page {page_num + 1}: {e}")
                    continue
                finally:
                    # Clean up temporary image
                    if os.path.exists(temp_image_path):
                        os.remove(temp_image_path)

            doc.close()
            logger.info("SmolDocLing OCR processing completed")
            return all_text.strip()

        except Exception as e:
            logger.error(f"SmolDocLing OCR extraction failed: {e}")
            return ""


class LLMClient:
    """Enhanced LLM client for entity extraction and validation"""

    def __init__(self, base_url: str = None, api_key: str = None, model: str = None):
        """
        Initialize LLM client

        Args:
            base_url: Base URL for the LLM service (vLLM or OpenAI)
            api_key: API key (optional for vLLM, required for OpenAI)
            model: Model name to use
        """
        # Load from environment if not provided
        self.base_url = base_url or os.getenv(
            "LLM_BASE_URL", "http://localhost:8000/v1"
        )
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.model = model or os.getenv("LLM_MODEL", "Qwen/Qwen3-8B")
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "2048"))
        self.temperature = float(
            os.getenv("LLM_TEMPERATURE", "0.1")
        )  # Lower temperature for more consistent extraction

        # Initialize OpenAI client if available and API key is provided
        self.client = None
        if OPENAI_AVAILABLE and self.api_key:
            try:
                openai.api_key = self.api_key
                openai.api_base = self.base_url
                self.client = openai.OpenAI(
                    api_key=self.api_key, base_url=self.base_url
                )
                logger.info(
                    f"LLM client initialized with OpenAI client - base_url: {self.base_url}, model: {self.model}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to initialize OpenAI client: {e}, falling back to HTTP requests"
                )
                self.client = None

        if self.client is None:
            logger.info(
                f"LLM client initialized with HTTP requests - base_url: {self.base_url}, model: {self.model}"
            )

    def health_check(self) -> bool:
        """Check if the LLM service is available"""
        try:
            if self.client:
                # Use OpenAI client for health check
                try:
                    if "localhost" in self.base_url or "127.0.0.1" in self.base_url:
                        response = self.client.get(
                            f"{self.base_url}/models", cast_to=Model
                        )
                    else:
                        response = self.client.models.list()
                    return True
                except Exception as e:
                    logger.debug(f"OpenAI client health check failed: {e}")
                    return False
            else:
                # Try health check endpoint for vLLM
                health_endpoints = f"{self.base_url}/models"

                try:
                    response = requests.get(health_endpoints, timeout=5)
                    if response.status_code == 200:
                        logger.debug(
                            f"Health check successful with endpoint: {health_endpoints}"
                        )
                        return True
                except Exception as e:
                    logger.debug(f"Health check failed for {health_endpoints}: {e}")
                    return False
        except Exception as e:
            logger.debug(f"Health check failed: {e}")
            return False

    def extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract structured entities from resume text

        Args:
            text: Resume text to extract entities from

        Returns:
            Dictionary containing extracted entities
        """
        if not self.health_check():
            logger.warning("LLM client not available for entity extraction")
            return {}

        try:
            # Create comprehensive extraction prompt
            prompt = self._create_entity_extraction_prompt(text)
            # Generate response
            response = self.generate(
                prompt, max_tokens=self.max_tokens, temperature=self.temperature
            )

            # Parse and validate response
            entities = self._parse_entity_response(response)

            # Validate extracted entities
            validated_entities = self._validate_entities(entities, text)

            return validated_entities

        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return {}

    def _create_entity_extraction_prompt(self, text: str) -> str:
        """Create comprehensive prompt for entity extraction"""
        return f"""You are an expert resume parser. Extract the following information from the resume text (which may be in markdown format) and return it as a valid JSON object.

Required fields to extract:
1. name: Full name of the candidate
2. email: Valid email address
3. phone: Phone number
4. skills: List of technical and professional skills
5. education: List of education entries with degree, institution, and graduation_year
6. experience: List of work experience with job_title, company, years_worked, and description
7. certifications: List of certifications
8. languages: List of languages the candidate can speak or write

Resume text (may contain markdown formatting):
{text}

Instructions:
- The text may contain markdown formatting (headers, lists, bold text, etc.)
- Pay attention to markdown headers (##) as they often indicate section boundaries
- Bold text (**text**) often indicates important information like names, titles, or skills
- Lists (- or *) often contain skills, education, or experience items
- Extract information regardless of the formatting

Return only a valid JSON object with the exact field names specified above. Do not include any additional text or explanations.

Example format:
{{
    "name": "John Doe",
    "email": "john.doe@example.com",
    "phone": "+1234567890",
    "skills": ["Python", "JavaScript", "React"],
    "education": [
        {{
            "degree": "Master of Science",
            "institution": "Stanford University",
            "graduation_year": "2020"
        }}
    ],
    "experience": [
        {{
            "job_title": "Software Engineer",
            "company": "TechCorp",
            "years_worked": "2020-2023",
            "description": "Developed web applications"
        }}
    ],
    "certifications": ["AWS Certified Developer"],
    "languages": ["English", "Spanish"]
}}"""

    def _parse_entity_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured entities"""
        try:
            # Try to extract JSON from response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                logger.warning("No JSON found in response, using fallback parsing")
                return self._fallback_entity_parsing(response)

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed: {e}, using fallback parsing")
            return self._fallback_entity_parsing(response)

    def _fallback_entity_parsing(self, response: str) -> Dict[str, Any]:
        """Fallback parsing when JSON parsing fails"""
        entities = {
            "name": "",
            "email": "",
            "phone": "",
            "skills": [],
            "education": [],
            "experience": [],
            "certifications": [],
            "languages": [],
        }

        # Simple keyword-based parsing
        lines = response.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Extract email
            if "@" in line and "." in line:
                email_match = line.split("@")[0] + "@" + line.split("@")[1].split()[0]
                if "@" in email_match and "." in email_match.split("@")[1]:
                    entities["email"] = email_match

            # Extract phone
            if any(char.isdigit() for char in line) and len(line) >= 10:
                # Simple phone extraction
                digits = "".join(filter(str.isdigit, line))
                if len(digits) >= 10:
                    entities["phone"] = line

        return entities

    def _validate_entities(
        self, entities: Dict[str, Any], original_text: str
    ) -> Dict[str, Any]:
        """Validate extracted entities against original text"""
        validated = entities.copy()

        # Validate email format
        if "email" in validated and validated["email"]:
            if not self._is_valid_email(validated["email"]):
                validated["email"] = ""

        # Validate phone format
        if "phone" in validated and validated["phone"]:
            if not self._is_valid_phone(validated["phone"]):
                validated["phone"] = ""

        # Ensure all required fields exist
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
        for field in required_fields:
            if field not in validated:
                if field in [
                    "skills",
                    "education",
                    "experience",
                    "certifications",
                    "languages",
                ]:
                    validated[field] = []
                else:
                    validated[field] = ""

        return validated

    def _is_valid_email(self, email: str) -> bool:
        """Validate email format"""
        import re

        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Z|a-z]{2,}$"
        return re.match(pattern, email) is not None

    def _is_valid_phone(self, phone: str) -> bool:
        """Validate phone number format"""
        digits = "".join(filter(str.isdigit, phone))
        return len(digits) >= 10

    def generate(
        self, prompt: str, max_tokens: int = None, temperature: float = None
    ) -> str:
        """Generate text using the configured LLM service"""
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature

        try:
            if self.client:
                # Use OpenAI client
                return self._generate_with_openai(prompt, max_tokens, temperature)
            else:
                # Use direct HTTP request (vLLM)
                return self._generate_with_http(prompt, max_tokens, temperature)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return ""

    def _generate_with_openai(
        self, prompt: str, max_tokens: int, temperature: float
    ) -> str:
        """Generate text using OpenAI client"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return response.choices[0].message.content.strip()

    def _generate_with_http(
        self, prompt: str, max_tokens: int, temperature: float
    ) -> str:
        """Generate text using direct HTTP request (vLLM)"""
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                timeout=30,
            )

            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"].strip()
                else:
                    raise Exception("No choices in response")
            else:
                error_msg = f"HTTP API error: {response.status_code}"
                try:
                    error_detail = response.json()
                    error_msg += f" - {error_detail}"
                except Exception as e:
                    logger.error(f"Error parsing response: {e}")
                    error_msg += f" - {response.text}"
                raise Exception(error_msg)
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {e}")
        except Exception as e:
            raise Exception(f"Generation failed: {e}")


class ResumeProcessor:
    """Main resume processor that combines SmolDocLing OCR and LLM entity extraction"""

    def __init__(
        self, llm_base_url: str = None, llm_api_key: str = None, llm_model: str = None
    ):
        self.smoldocling_client = SmolDocLingClient()
        self.llm_client = LLMClient(llm_base_url, llm_api_key, llm_model)

    def process_resume(self, pdf_path: str) -> Dict[str, Any]:
        """
        Process resume PDF and extract structured entities

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary containing extracted entities and metadata
        """
        try:
            logger.info(f"Processing resume: {pdf_path}")

            # Step 1: Extract text using SmolDocLing OCR
            extracted_text = self.smoldocling_client.extract_text_from_pdf(pdf_path)

            if not extracted_text.strip():
                logger.error("No text extracted from PDF")
                return {"error": "No text extracted from PDF"}

            # Step 2: Extract entities using LLM
            entities = self.llm_client.extract_entities(extracted_text)

            # Step 3: Add metadata
            result = {
                "entities": entities,
                "metadata": {
                    "file_path": pdf_path,
                    "file_size": os.path.getsize(pdf_path),
                    "text_length": len(extracted_text),
                    "processing_method": "smoldocling_ocr_llm_extraction",
                    "smoldocling_available": self.smoldocling_client.health_check(),
                    "llm_available": self.llm_client.health_check(),
                    "extracted_text_preview": extracted_text[:500] + "..."
                    if len(extracted_text) > 500
                    else extracted_text,
                },
            }

            logger.info(f"Successfully processed resume: {pdf_path}")
            return result

        except Exception as e:
            logger.error(f"Error processing resume {pdf_path}: {e}")
            return {"error": str(e)}

    def validate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate extracted results

        Args:
            results: Results from process_resume

        Returns:
            Validation results
        """
        validation = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "completeness_score": 0.0,
        }

        if "error" in results:
            validation["is_valid"] = False
            validation["errors"].append(results["error"])
            return validation

        entities = results.get("entities", {})
        required_fields = [
            "name",
            "email",
            "phone",
            "skills",
            "education",
            "experience",
        ]

        # Check completeness
        present_fields = 0
        for field in required_fields:
            if field in entities and entities[field]:
                if isinstance(entities[field], list):
                    if len(entities[field]) > 0:
                        present_fields += 1
                else:
                    if entities[field].strip():
                        present_fields += 1

        validation["completeness_score"] = present_fields / len(required_fields)

        # Check for warnings
        if validation["completeness_score"] < 0.5:
            validation["warnings"].append("Low completeness score")

        if not entities.get("email"):
            validation["warnings"].append("No email found")

        if not entities.get("phone"):
            validation["warnings"].append("No phone found")

        return validation


# Factory functions for easy client creation
def create_smoldocling_client(
    model_name: str = "ds4sd/SmolDocling-256M-preview",
) -> SmolDocLingClient:
    """Create a SmolDocLing client"""
    return SmolDocLingClient(model_name)


def create_llm_client(
    base_url: str = None, api_key: str = None, model: str = None
) -> LLMClient:
    """Create an LLM client"""
    return LLMClient(base_url, api_key, model)


def create_resume_processor(
    llm_base_url: str = None, llm_api_key: str = None, llm_model: str = None
) -> ResumeProcessor:
    """Create a resume processor with both SmolDocLing and LLM clients"""
    return ResumeProcessor(llm_base_url, llm_api_key, llm_model)
