#!/usr/bin/env python3
"""
Resume Parser API Usage Examples

This script demonstrates how to use the Resume Parser API with various endpoints
and different file formats. It includes error handling and best practices.
"""

import requests
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# API Configuration
API_BASE_URL = "http://localhost:8000"
API_ENDPOINTS = {
    "parse": f"{API_BASE_URL}/parse",
    "batch": f"{API_BASE_URL}/parse/batch",
    "specific": f"{API_BASE_URL}/extract/specific",
    "health": f"{API_BASE_URL}/health",
    "vllm_status": f"{API_BASE_URL}/vllm/status"
}

class ResumeParserAPI:
    """Client for interacting with the Resume Parser API"""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ResumeParserAPI/1.0'
        })
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health status"""
        try:
            response = self.session.get(API_ENDPOINTS["health"])
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"Health check failed: {e}"}
    
    def vllm_status(self) -> Dict[str, Any]:
        """Check vLLM server status"""
        try:
            response = self.session.get(API_ENDPOINTS["vllm_status"])
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"vLLM status check failed: {e}"}
    
    def parse_single_resume(self, file_path: str, use_llm: bool = True) -> Dict[str, Any]:
        """Parse a single resume file"""
        try:
            if not os.path.exists(file_path):
                return {"error": f"File not found: {file_path}"}
            
            with open(file_path, 'rb') as file:
                files = {'file': (os.path.basename(file_path), file, 'application/octet-stream')}
                data = {'use_llm': str(use_llm).lower()}
                
                response = self.session.post(
                    API_ENDPOINTS["parse"],
                    files=files,
                    data=data,
                    timeout=120  # 2 minutes timeout
                )
                response.raise_for_status()
                return response.json()
                
        except requests.exceptions.RequestException as e:
            return {"error": f"API request failed: {e}"}
        except Exception as e:
            return {"error": f"Unexpected error: {e}"}
    
    def parse_batch_resumes(self, file_paths: List[str], use_llm: bool = True) -> Dict[str, Any]:
        """Parse multiple resume files in batch"""
        try:
            files = []
            for file_path in file_paths:
                if not os.path.exists(file_path):
                    return {"error": f"File not found: {file_path}"}
                
                with open(file_path, 'rb') as file:
                    files.append(('files', (os.path.basename(file_path), file, 'application/octet-stream')))
            
            data = {'use_llm': str(use_llm).lower()}
            
            response = self.session.post(
                API_ENDPOINTS["batch"],
                files=files,
                data=data,
                timeout=300  # 5 minutes timeout for batch processing
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            return {"error": f"Batch API request failed: {e}"}
        except Exception as e:
            return {"error": f"Unexpected error: {e}"}
    
    def extract_specific_info(self, file_path: str, info_type: str) -> Dict[str, Any]:
        """Extract specific information from resume"""
        try:
            if not os.path.exists(file_path):
                return {"error": f"File not found: {file_path}"}
            
            with open(file_path, 'rb') as file:
                files = {'file': (os.path.basename(file_path), file, 'application/octet-stream')}
                data = {'info_type': info_type}
                
                response = self.session.post(
                    API_ENDPOINTS["specific"],
                    files=files,
                    data=data,
                    timeout=60
                )
                response.raise_for_status()
                return response.json()
                
        except requests.exceptions.RequestException as e:
            return {"error": f"Specific extraction failed: {e}"}
        except Exception as e:
            return {"error": f"Unexpected error: {e}"}

def print_result(result: Dict[str, Any], title: str = "Result"):
    """Pretty print API results"""
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return
    
    # Print structured information
    if "personal_info" in result:
        print("👤 Personal Information:")
        for key, value in result["personal_info"].items():
            print(f"   {key}: {value}")
    
    if "education" in result:
        print("\n🎓 Education:")
        for edu in result["education"]:
            print(f"   - {edu.get('degree', 'N/A')} from {edu.get('institution', 'N/A')}")
    
    if "experience" in result:
        print("\n💼 Work Experience:")
        for exp in result["experience"]:
            print(f"   - {exp.get('title', 'N/A')} at {exp.get('company', 'N/A')}")
    
    if "skills" in result:
        print(f"\n🔧 Skills ({len(result['skills'])}):")
        print(f"   {', '.join(result['skills'])}")
    
    if "summary" in result:
        print(f"\n📝 Summary:")
        print(f"   {result['summary']}")
    
    if "llm_summary" in result:
        print(f"\n🤖 LLM Summary:")
        print(f"   {result['llm_summary']}")
    
    if "key_achievements" in result:
        print(f"\n🏆 Key Achievements:")
        for achievement in result["key_achievements"]:
            print(f"   - {achievement}")

def example_1_basic_parsing():
    """Example 1: Basic resume parsing"""
    print("\n📋 Example 1: Basic Resume Parsing")
    print("-" * 40)
    
    api = ResumeParserAPI()
    
    # Check API health first
    health = api.health_check()
    print(f"API Health: {health}")
    
    # Parse a single resume
    result = api.parse_single_resume("examples/sample_resume.pdf")
    print_result(result, "Basic Resume Parsing Result")

def example_2_batch_processing():
    """Example 2: Batch processing multiple resumes"""
    print("\n📋 Example 2: Batch Resume Processing")
    print("-" * 40)
    
    api = ResumeParserAPI()
    
    # List of resume files to process
    resume_files = [
        "examples/resume1.pdf",
        "examples/resume2.docx",
        "examples/resume3.txt"
    ]
    
    # Check which files exist
    existing_files = [f for f in resume_files if os.path.exists(f)]
    if not existing_files:
        print("❌ No sample files found. Please add some resume files to the examples/ directory.")
        return
    
    print(f"Processing {len(existing_files)} resume files...")
    result = api.parse_batch_resumes(existing_files)
    
    if "error" not in result:
        print(f"\n✅ Batch processing completed!")
        print(f"Processed {len(result.get('results', []))} resumes")
        
        # Print summary
        if "summary" in result:
            print(f"Batch Summary: {result['summary']}")
    else:
        print(f"❌ Batch processing failed: {result['error']}")

def example_3_specific_extraction():
    """Example 3: Extract specific information"""
    print("\n📋 Example 3: Specific Information Extraction")
    print("-" * 40)
    
    api = ResumeParserAPI()
    
    # Extract different types of information
    info_types = ["skills", "experience", "education", "contact"]
    
    for info_type in info_types:
        print(f"\n🔍 Extracting {info_type}...")
        result = api.extract_specific_info("examples/sample_resume.pdf", info_type)
        
        if "error" not in result:
            print(f"✅ {info_type.title()} extracted successfully")
            if "extracted_info" in result:
                print(f"   Found: {result['extracted_info']}")
        else:
            print(f"❌ Failed to extract {info_type}: {result['error']}")

def example_4_without_llm():
    """Example 4: Parsing without LLM enhancement"""
    print("\n📋 Example 4: Parsing Without LLM Enhancement")
    print("-" * 40)
    
    api = ResumeParserAPI()
    
    # Parse without LLM enhancement
    result = api.parse_single_resume("examples/sample_resume.pdf", use_llm=False)
    print_result(result, "Parsing Without LLM")

def example_5_error_handling():
    """Example 5: Error handling examples"""
    print("\n📋 Example 5: Error Handling Examples")
    print("-" * 40)
    
    api = ResumeParserAPI()
    
    # Test with non-existent file
    print("Testing with non-existent file...")
    result = api.parse_single_resume("non_existent_file.pdf")
    print_result(result, "Non-existent File Test")
    
    # Test with unsupported file type
    print("\nTesting with unsupported file type...")
    result = api.parse_single_resume("examples/test.txt")
    print_result(result, "Unsupported File Type Test")

def example_6_vllm_status():
    """Example 6: Check vLLM server status"""
    print("\n📋 Example 6: vLLM Server Status")
    print("-" * 40)
    
    api = ResumeParserAPI()
    
    # Check vLLM status
    status = api.vllm_status()
    print(f"vLLM Status: {json.dumps(status, indent=2)}")

def create_sample_files():
    """Create sample files for testing"""
    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)
    
    # Create a sample text resume
    sample_text = """
    JOHN DOE
    Software Engineer
    john.doe@email.com | (555) 123-4567 | linkedin.com/in/johndoe
    
    SUMMARY
    Experienced software engineer with 5+ years in full-stack development.
    
    EXPERIENCE
    Senior Developer | Tech Corp | 2020-2023
    - Led development of web applications
    - Mentored junior developers
    
    Junior Developer | Startup Inc | 2018-2020
    - Developed REST APIs
    - Implemented CI/CD pipelines
    
    EDUCATION
    Bachelor of Science in Computer Science
    University of Technology | 2018
    
    SKILLS
    Python, JavaScript, React, Node.js, Docker, AWS
    """
    
    with open(examples_dir / "sample_resume.txt", "w") as f:
        f.write(sample_text)
    
    print("✅ Created sample resume file: examples/sample_resume.txt")

def main():
    """Main function to run all examples"""
    print("🚀 Resume Parser API Usage Examples")
    print("=" * 50)
    
    # Create sample files if they don't exist
    if not os.path.exists("examples/sample_resume.txt"):
        create_sample_files()
    
    # Run examples
    try:
        example_1_basic_parsing()
        example_2_batch_processing()
        example_3_specific_extraction()
        example_4_without_llm()
        example_5_error_handling()
        example_6_vllm_status()
        
        print("\n✅ All examples completed!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main() 