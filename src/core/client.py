#!/usr/bin/env python3
"""
Unified LLM Client
Treats vLLM and OpenAI as the same service with different base URLs
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI imports
try:
    import openai
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
    SMOLDOCLING_AVAILABLE = True
except ImportError:
    SMOLDOCLING_AVAILABLE = False

logger = logging.getLogger(__name__)

class LLMClient:
    """Unified LLM client that treats vLLM and OpenAI as the same service"""
    
    def __init__(self, base_url: str = None, api_key: str = None, model: str = None):
        """
        Initialize LLM client
        
        Args:
            base_url: Base URL for the LLM service (vLLM or OpenAI)
            api_key: API key (optional for vLLM, required for OpenAI)
            model: Model name to use
        """
        # Load from environment if not provided
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "http://localhost:8000")
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.model = model or os.getenv("LLM_MODEL", "Qwen/Qwen3-8B")
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "1024"))
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        
        # Initialize OpenAI client if available
        if OPENAI_AVAILABLE and self.api_key:
            openai.api_key = self.api_key
            openai.api_base = self.base_url
            self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
            logger.info(f"LLM client initialized with base_url: {self.base_url}, model: {self.model}")
        else:
            self.client = None
            logger.info(f"LLM client initialized with base_url: {self.base_url}, model: {self.model}")
    
    def health_check(self) -> bool:
        """Check if the LLM service is available"""
        try:
            if self.client:
                # Use OpenAI client for health check
                response = self.client.models.list()
                return True
            else:
                # Use direct HTTP request for vLLM
                response = requests.get(f"{self.base_url}/v1/models", timeout=5)
                return response.status_code == 200
        except Exception as e:
            logger.debug(f"Health check failed: {e}")
            return False
    
    def generate(self, prompt: str, max_tokens: int = None, temperature: float = None) -> str:
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
    
    def _generate_with_openai(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate text using OpenAI client"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    
    def _generate_with_http(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate text using direct HTTP request (vLLM)"""
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        else:
            raise Exception(f"HTTP API error: {response.status_code}")
    
    def extract_specific_info(self, text: str, info_type: str) -> str:
        """Extract specific information from text"""
        prompts = {
            'skills': f"Extract technical skills and programming languages from this text:\n{text}",
            'experience': f"Extract work experience and job titles from this text:\n{text}",
            'education': f"Extract education and degrees from this text:\n{text}",
            'contact': f"Extract contact information (email, phone, linkedin) from this text:\n{text}",
            'summary': f"Write a brief professional summary for this resume:\n{text}",
            'achievements': f"Extract key achievements and accomplishments from this text:\n{text}"
        }
        
        prompt = prompts.get(info_type, f"Extract {info_type} from this text:\n{text}")
        return self.generate(prompt, max_tokens=500, temperature=0.3)

class SmolDocLingOCR:
    """SmolDocLing OCR implementation for document processing"""
    
    def __init__(self, model_name: str = "ds4sd/SmolDocling-256M-preview"):
        self.model_name = model_name
        self.processor = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if SMOLDOCLING_AVAILABLE:
            try:
                self.processor = AutoProcessor.from_pretrained(model_name)
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    _attn_implementation="eager",  # Use eager to avoid FlashAttention2 dependency
                ).to(self.device)
                logger.info(f"SmolDocLing initialized with model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize SmolDocLing: {e}")
        else:
            logger.warning("SmolDocLing packages not available")
    
    def health_check(self) -> bool:
        """Check if SmolDocLing is available"""
        return self.processor is not None and self.model is not None
    
    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from image using SmolDocLing"""
        if not self.health_check():
            return ""
        
        try:
            # Load image
            image = load_image(image_path)
            
            # Create input messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Convert this page to docling."}
                    ]
                },
            ]
            
            # Prepare inputs
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
            inputs = inputs.to(self.device)
            
            # Generate outputs
            generated_ids = self.model.generate(**inputs, max_new_tokens=8192)
            prompt_length = inputs.input_ids.shape[1]
            trimmed_generated_ids = generated_ids[:, prompt_length:]
            doctags = self.processor.batch_decode(
                trimmed_generated_ids,
                skip_special_tokens=False,
            )[0].lstrip()
            
            # Convert to DoclingDocument and extract text
            doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
            doc = DoclingDocument.load_from_doctags(doctags_doc, document_name="Document")
            
            return doc.export_to_markdown()
            
        except Exception as e:
            logger.error(f"SmolDocLing OCR failed: {e}")
            return ""
    
    def extract_text_from_pdf_pages(self, pdf_path: str) -> str:
        """Extract text from PDF pages using SmolDocLing"""
        if not SMOLDOCLING_AVAILABLE:
            return ""
        
        try:
            import fitz  # PyMuPDF
            
            # Open PDF
            doc = fitz.open(pdf_path)
            all_text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Convert page to image
                pix = page.get_pixmap()
                img_data = pix.tobytes("png")
                
                # Save temporary image
                temp_image_path = f"/tmp/page_{page_num}.png"
                with open(temp_image_path, "wb") as f:
                    f.write(img_data)
                
                # Extract text from image
                page_text = self.extract_text_from_image(temp_image_path)
                all_text += page_text + "\n\n"
                
                # Clean up
                os.remove(temp_image_path)
            
            doc.close()
            return all_text.strip()
            
        except Exception as e:
            logger.error(f"SmolDocLing PDF processing failed: {e}")
            return ""

class ResumeLLMProcessor:
    """Enhanced resume processing using unified LLM client and SmolDocLing OCR"""
    
    def __init__(self, base_url: str = None, api_key: str = None, model: str = None):
        self.llm_client = LLMClient(base_url, api_key, model)
        self.ocr_processor = SmolDocLingOCR()
        
    def enhance_extraction(self, structured_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance extracted data using the configured LLM provider"""
        if not self.llm_client.health_check():
            logger.warning("LLM client not available, skipping LLM enhancement")
            return structured_data
        
        try:
            # Create prompt for LLM enhancement
            prompt = self._create_enhancement_prompt(structured_data)
            
            # Get LLM response
            enhanced_info = self.llm_client.generate(prompt, max_tokens=1024)
            
            # Parse enhanced information
            enhanced_data = self._parse_llm_response(enhanced_info)
            
            # Merge with original data
            enhanced_data.update(structured_data)
            
            return enhanced_data
            
        except Exception as e:
            logger.error(f"Error in LLM enhancement: {e}")
            return structured_data
    
    def _create_enhancement_prompt(self, data: Dict[str, Any]) -> str:
        """Create prompt for resume enhancement"""
        raw_text = data.get('raw_text', '')
        
        prompt = f"""You are an expert resume parser. Analyze the following resume text and extract structured information in JSON format. Be precise and accurate in your extraction.

Please analyze this resume and extract the following information in valid JSON format:

{raw_text}

Extract:
1. Personal Information (name, email, phone, location)
2. Education (degree, institution, year, GPA if available)
3. Work Experience (company, position, duration, key achievements)
4. Skills (technical skills, soft skills, certifications)
5. Summary/Objective
6. Contact Information (email, phone, linkedin, portfolio)

Return only valid JSON without any additional text or explanations."""
        
        return prompt
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured data"""
        try:
            # Try to extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                # Fallback parsing
                return self._fallback_parsing(response)
                
        except json.JSONDecodeError:
            return self._fallback_parsing(response)
    
    def _fallback_parsing(self, response: str) -> Dict[str, Any]:
        """Fallback parsing when JSON parsing fails"""
        parsed_data = {}
        
        # Simple keyword-based parsing
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Detect sections
            if 'personal' in line.lower() or 'name' in line.lower():
                current_section = 'personal_info'
                parsed_data[current_section] = {}
            elif 'education' in line.lower():
                current_section = 'education'
                parsed_data[current_section] = []
            elif 'experience' in line.lower() or 'work' in line.lower():
                current_section = 'experience'
                parsed_data[current_section] = []
            elif 'skills' in line.lower():
                current_section = 'skills'
                parsed_data[current_section] = []
            elif 'contact' in line.lower():
                current_section = 'contact'
                parsed_data[current_section] = {}
            
            # Parse content based on current section
            if current_section and ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                if current_section == 'personal_info':
                    parsed_data[current_section][key] = value
                elif current_section in ['education', 'experience', 'skills']:
                    parsed_data[current_section].append({key: value})
                elif current_section == 'contact':
                    parsed_data[current_section][key] = value
        
        return parsed_data
    
    def extract_specific_info(self, text: str, info_type: str) -> str:
        """Extract specific information using the configured LLM provider"""
        if not self.llm_client.health_check():
            return ""
        
        return self.llm_client.extract_specific_info(text, info_type)
    
    def summarize_resume(self, text: str) -> str:
        """Generate a summary of the resume using the configured LLM provider"""
        if not self.llm_client.health_check():
            return ""
        
        prompt = f"""You are an expert at creating professional summaries of resumes.

Provide a brief professional summary of this resume: {text[:1000]}"""
        
        return self.llm_client.generate(prompt, max_tokens=200)
    
    def extract_key_achievements(self, text: str) -> List[str]:
        """Extract key achievements from resume using the configured LLM provider"""
        if not self.llm_client.health_check():
            return []
        
        prompt = f"""You are an expert at extracting key achievements and accomplishments from resumes.

Extract key achievements and accomplishments from this resume text: {text}"""
        
        response = self.llm_client.generate(prompt, max_tokens=512)
        
        # Parse achievements (assuming they're separated by newlines or bullets)
        achievements = []
        for line in response.split('\n'):
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('•') or line.startswith('*')):
                achievements.append(line[1:].strip())
            elif line and len(line) > 10:
                achievements.append(line)
        
        return achievements[:10]  # Limit to 10 achievements
    
    def extract_text_with_smoldocling(self, file_path: str) -> str:
        """Extract text from document using SmolDocLing OCR"""
        if not self.ocr_processor.health_check():
            logger.warning("SmolDocLing not available, skipping OCR extraction")
            return ""
        
        try:
            # Check if it's a PDF
            if file_path.lower().endswith('.pdf'):
                return self.ocr_processor.extract_text_from_pdf_pages(file_path)
            else:
                # Assume it's an image
                return self.ocr_processor.extract_text_from_image(file_path)
        except Exception as e:
            logger.error(f"SmolDocLing extraction failed: {e}")
            return ""

# Factory function for easy client creation
def create_llm_client(base_url: str = None, api_key: str = None, model: str = None) -> LLMClient:
    """Create an LLM client with the specified configuration"""
    return LLMClient(base_url, api_key, model) 