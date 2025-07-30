import requests
import json
import logging
from typing import Dict, List, Any, Optional
from config import vllm_config

logger = logging.getLogger(__name__)

class VLLMClient:
    """Client for interacting with vLLM server hosting Qwen3-8B"""
    
    def __init__(self, base_url: str = None):
        if base_url is None:
            base_url = f"http://{vllm_config.host}:{vllm_config.port}"
        self.base_url = base_url
        self.session = requests.Session()
        
    def health_check(self) -> bool:
        """Check if vLLM server is running"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Generate text using vLLM server with Qwen3-8B"""
        try:
            payload = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop": ["<|im_end|>", "\n\n", "###", "END"]
            }
            
            response = self.session.post(
                f"{self.base_url}/v1/completions",
                json=payload,
                timeout=60  # Increased timeout for Qwen
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("choices", [{}])[0].get("text", "").strip()
            else:
                logger.error(f"Generation failed: {response.status_code} - {response.text}")
                return ""
                
        except Exception as e:
            logger.error(f"Error in text generation: {e}")
            return ""
    
    def chat_completion(self, messages: List[Dict[str, str]], max_tokens: int = 512) -> str:
        """Generate chat completion using vLLM server with Qwen3-8B"""
        try:
            payload = {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.7
            }
            
            response = self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=60  # Increased timeout for Qwen
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            else:
                logger.error(f"Chat completion failed: {response.status_code} - {response.text}")
                return ""
                
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            return ""

class ResumeLLMProcessor:
    """Enhanced resume processing using Qwen3-8B LLM"""
    
    def __init__(self):
        self.llm_client = VLLMClient()
        
    def enhance_extraction(self, structured_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance extracted data using Qwen3-8B"""
        if not self.llm_client.health_check():
            logger.warning("vLLM server not available, skipping LLM enhancement")
            return structured_data
        
        try:
            # Create prompt for LLM enhancement using Qwen format
            prompt = self._create_qwen_enhancement_prompt(structured_data)
            
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
    
    def _create_qwen_enhancement_prompt(self, data: Dict[str, Any]) -> str:
        """Create Qwen-formatted prompt for resume enhancement"""
        raw_text = data.get('raw_text', '')
        
        prompt = f"""<|im_start|>system
You are an expert resume parser. Analyze the following resume text and extract structured information in JSON format. Be precise and accurate in your extraction.
<|im_end|>
<|im_start|>user
Please analyze this resume and extract the following information in valid JSON format:

{raw_text}

Extract:
1. Personal Information (name, email, phone, location)
2. Education (degree, institution, year, GPA if available)
3. Work Experience (company, position, duration, key achievements)
4. Skills (technical skills, soft skills, certifications)
5. Summary/Objective
6. Contact Information (email, phone, linkedin, portfolio)

Return only valid JSON without any additional text or explanations.
<|im_end|>
<|im_start|>assistant
"""
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
        """Extract specific information using Qwen3-8B"""
        if not self.llm_client.health_check():
            return ""
        
        prompts = {
            'skills': f"""<|im_start|>system
You are an expert at extracting technical skills and programming languages from resumes.
<|im_end|>
<|im_start|>user
Extract technical skills and programming languages from this text: {text}
<|im_end|>
<|im_start|>assistant
""",
            'experience': f"""<|im_start|>system
You are an expert at extracting work experience and job titles from resumes.
<|im_end|>
<|im_start|>user
Extract work experience and job titles from this text: {text}
<|im_end|>
<|im_start|>assistant
""",
            'education': f"""<|im_start|>system
You are an expert at extracting education and degrees from resumes.
<|im_end|>
<|im_start|>user
Extract education and degrees from this text: {text}
<|im_end|>
<|im_start|>assistant
""",
            'contact': f"""<|im_start|>system
You are an expert at extracting contact information from resumes.
<|im_end|>
<|im_start|>user
Extract contact information (email, phone, linkedin) from this text: {text}
<|im_end|>
<|im_start|>assistant
"""
        }
        
        if info_type not in prompts:
            return ""
        
        return self.llm_client.generate(prompts[info_type], max_tokens=256)
    
    def summarize_resume(self, text: str) -> str:
        """Generate a summary of the resume using Qwen3-8B"""
        if not self.llm_client.health_check():
            return ""
        
        prompt = f"""<|im_start|>system
You are an expert at creating professional summaries of resumes.
<|im_end|>
<|im_start|>user
Provide a brief professional summary of this resume: {text[:1000]}
<|im_end|>
<|im_start|>assistant
"""
        return self.llm_client.generate(prompt, max_tokens=200)
    
    def extract_key_achievements(self, text: str) -> List[str]:
        """Extract key achievements from resume using Qwen3-8B"""
        if not self.llm_client.health_check():
            return []
        
        prompt = f"""<|im_start|>system
You are an expert at extracting key achievements and accomplishments from resumes.
<|im_end|>
<|im_start|>user
Extract key achievements and accomplishments from this resume text: {text}
<|im_end|>
<|im_start|>assistant
"""
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