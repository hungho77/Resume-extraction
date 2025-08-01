#!/usr/bin/env python3
"""
Prompt templates for LLM interactions
Centralized prompt management for clean code organization
"""

from typing import Dict, Any


class ResumeExtractionPrompts:
    """Prompt templates for resume entity extraction"""
    
    @staticmethod
    def entity_extraction_prompt(text: str) -> str:
        """Create comprehensive prompt for entity extraction"""
        return f"""You are an assistant specialized in resume parsing and entity extraction with 10+ years of experience in NLP and document processing. 
Your task is to accurately extract structured information from the provided resume text, which may be in markdown format. Pay special attention to extracting the candidate's full name correctly—it is typically the first prominent text at the top of the resume, often in a header (e.g., # Name or **Name**), and not to be confused with company names, job titles, or other bold elements later in the document.
If no clear personal name (e.g., 'First Last') is found explicitly at the top, set it to an empty string—do not use job titles, companies, or infer from context. 
Extract the following information from the resume text (which may be in markdown format) and return it as a valid JSON object.

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

Output format:
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

    @staticmethod
    def specific_field_extraction_prompt(text: str, field: str) -> str:
        """Create prompt for extracting a specific field from resume"""
        field_descriptions = {
            "name": "full name of the candidate",
            "email": "valid email address",
            "phone": "phone number",
            "skills": "list of technical and professional skills",
            "education": "list of education entries with degree, institution, and graduation_year",
            "experience": "list of work experience with job_title, company, years_worked, and description",
            "certifications": "list of certifications",
            "languages": "list of languages the candidate can speak or write"
        }
        
        field_desc = field_descriptions.get(field, field)
        
        return f"""You are an expert resume parser. Extract only the {field_desc} from the resume text (which may be in markdown format) and return it as a valid JSON object.

Resume text (may contain markdown formatting):
{text}

Instructions:
- The text may contain markdown formatting (headers, lists, bold text, etc.)
- Pay attention to markdown headers (##) as they often indicate section boundaries
- Bold text (**text**) often indicates important information like names, titles, or skills
- Lists (- or *) often contain skills, education, or experience items
- Extract only the {field} information

Return only a valid JSON object with the {field} field. Do not include any additional text or explanations.

Example format for {field}:
{{
    "{field}": {_get_field_example(field)}
}}"""


class ValidationPrompts:
    """Prompt templates for validation tasks"""
    
    @staticmethod
    def entity_validation_prompt(entities: Dict[str, Any], original_text: str) -> str:
        """Create prompt for validating extracted entities"""
        return f"""You are an expert resume validator. Review the extracted entities and validate their accuracy and completeness.

Extracted entities:
{entities}

Original resume text:
{original_text}

Please validate:
1. Completeness: Are all required fields present and filled?
2. Accuracy: Do the extracted entities match the information in the original text?
3. Format: Are emails and phone numbers in correct format?
4. Consistency: Are there any contradictions or inconsistencies?

Return a JSON object with validation results:
{{
    "is_valid": true/false,
    "completeness_score": 0.0-1.0,
    "accuracy_score": 0.0-1.0,
    "errors": ["list of specific errors found"],
    "warnings": ["list of warnings"],
    "suggestions": ["list of improvement suggestions"]
}}"""


class SystemPrompts:
    """System-level prompt templates"""
    
    @staticmethod
    def resume_parser_system_prompt() -> str:
        """System prompt for resume parsing tasks"""
        return """You are an expert resume parser with deep understanding of:
- Professional resume formats and structures
- Technical skills and certifications
- Educational qualifications and institutions
- Work experience and job titles
- Contact information validation
- Markdown formatting interpretation

Your task is to extract structured information from resumes accurately and completely."""


def _get_field_example(field: str) -> str:
    """Get example format for specific fields"""
    examples = {
        "name": '"John Doe"',
        "email": '"john.doe@example.com"',
        "phone": '"+1234567890"',
        "skills": '["Python", "JavaScript", "React"]',
        "education": """[
            {
                "degree": "Master of Science",
                "institution": "Stanford University", 
                "graduation_year": "2020"
            }
        ]""",
        "experience": """[
            {
                "job_title": "Software Engineer",
                "company": "TechCorp",
                "years_worked": "2020-2023",
                "description": "Developed web applications"
            }
        ]""",
        "certifications": '["AWS Certified Developer"]',
        "languages": '["English", "Spanish"]'
    }
    return examples.get(field, '""')


# Convenience functions for easy access
def get_entity_extraction_prompt(text: str) -> str:
    """Get entity extraction prompt"""
    return ResumeExtractionPrompts.entity_extraction_prompt(text)


def get_specific_field_prompt(text: str, field: str) -> str:
    """Get specific field extraction prompt"""
    return ResumeExtractionPrompts.specific_field_extraction_prompt(text, field)


def get_validation_prompt(entities: Dict[str, Any], original_text: str) -> str:
    """Get validation prompt"""
    return ValidationPrompts.entity_validation_prompt(entities, original_text)


def get_system_prompt() -> str:
    """Get system prompt"""
    return SystemPrompts.resume_parser_system_prompt()
