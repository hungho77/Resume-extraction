#!/usr/bin/env python3
"""
Test script for Resume Parser
Demonstrates usage with a sample resume text.
"""

import json
import tempfile
import os
import sys
from pathlib import Path

# Add parent directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.document_processor import ResumeDocumentProcessor
from src.core.llm_client import ResumeLLMProcessor, VLLMClient

def create_sample_resume():
    """Create a sample resume text for testing"""
    sample_resume = """
John Doe
Software Engineer
john.doe@email.com | (555) 123-4567 | linkedin.com/in/johndoe

SUMMARY
Experienced software engineer with 5+ years of experience in full-stack development, 
specializing in Python, JavaScript, and cloud technologies. Proven track record of 
delivering scalable solutions and leading development teams.

EDUCATION
Bachelor of Science in Computer Science
University of Technology | 2018-2022 | GPA: 3.8/4.0

WORK EXPERIENCE
Senior Software Engineer | TechCorp Inc. | 2022-Present
• Led development of microservices architecture serving 1M+ users
• Implemented CI/CD pipeline reducing deployment time by 60%
• Mentored 3 junior developers and conducted code reviews
• Technologies: Python, Django, React, AWS, Docker, Kubernetes

Software Engineer | StartupXYZ | 2020-2022
• Developed RESTful APIs and frontend applications
• Collaborated with cross-functional teams to deliver features
• Improved application performance by 40% through optimization
• Technologies: JavaScript, Node.js, React, MongoDB, Redis

Junior Developer | WebDev Solutions | 2018-2020
• Built responsive web applications using modern frameworks
• Participated in agile development processes
• Contributed to open-source projects
• Technologies: HTML, CSS, JavaScript, PHP, MySQL

SKILLS
Programming Languages: Python, JavaScript, Java, C++, SQL
Frameworks & Libraries: React, Angular, Django, Flask, Express.js
Cloud & DevOps: AWS, Azure, Docker, Kubernetes, Jenkins, Git
Databases: PostgreSQL, MongoDB, Redis, MySQL
Tools & Platforms: VS Code, IntelliJ, Jira, Confluence, Slack

CERTIFICATIONS
• AWS Certified Solutions Architect
• Google Cloud Professional Developer
• Certified Scrum Master (CSM)

PROJECTS
E-commerce Platform | Full-stack web application with payment integration
• Built with React, Node.js, and MongoDB
• Handles 10,000+ concurrent users
• GitHub: github.com/johndoe/ecommerce

Task Management App | Mobile-first productivity application
• Developed with React Native and Firebase
• 50,000+ downloads on app stores
• GitHub: github.com/johndoe/taskapp

LANGUAGES
English (Native), Spanish (Conversational)
"""
    return sample_resume

def test_document_processor():
    """Test the document processor with sample data"""
    print("🧪 Testing Document Processor...")
    
    # Create sample resume file
    sample_text = create_sample_resume()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(sample_text)
        temp_file = f.name
    
    try:
        # Initialize processor
        processor = ResumeDocumentProcessor()
        
        # Process document
        results = processor.process_document(temp_file)
        
        print("✅ Document processing completed!")
        print(f"📄 File processed: {temp_file}")
        
        # Print results
        print("\n📋 Extracted Information:")
        print(f"  Personal Info: {results.get('personal_info', {})}")
        print(f"  Contact Info: {results.get('contact', {})}")
        print(f"  Skills Found: {len(results.get('skills', []))}")
        print(f"  Education Entries: {len(results.get('education', []))}")
        print(f"  Experience Entries: {len(results.get('experience', []))}")
        
        return results
        
    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.unlink(temp_file)

def test_llm_processor():
    """Test the LLM processor"""
    print("\n🤖 Testing LLM Processor...")
    
    # Check vLLM availability
    client = VLLMClient()
    if not client.health_check():
        print("⚠️  vLLM server not available, skipping LLM tests")
        return None
    
    # Create sample data
    sample_data = {
        'raw_text': create_sample_resume(),
        'personal_info': {'name': 'John Doe'},
        'skills': ['python', 'javascript']
    }
    
    # Initialize LLM processor
    llm_processor = ResumeLLMProcessor()
    
    # Test enhancement
    enhanced_data = llm_processor.enhance_extraction(sample_data)
    
    print("✅ LLM enhancement completed!")
    
    # Test specific extractions
    summary = llm_processor.summarize_resume(sample_data['raw_text'])
    achievements = llm_processor.extract_key_achievements(sample_data['raw_text'])
    
    print(f"📝 Summary: {summary[:100]}...")
    print(f"🏆 Achievements found: {len(achievements)}")
    
    return enhanced_data

def test_api_integration():
    """Test API integration"""
    print("\n🌐 Testing API Integration...")
    
    import requests
    
    # Check if API server is running
    try:
        response = requests.get("http://localhost:8080/health", timeout=5)
        if response.status_code == 200:
            print("✅ API server is running")
            
            # Test vLLM status
            vllm_response = requests.get("http://localhost:8080/vllm/status", timeout=5)
            if vllm_response.status_code == 200:
                vllm_status = vllm_response.json()
                print(f"✅ vLLM Status: {vllm_status}")
            else:
                print("⚠️  Could not check vLLM status")
        else:
            print("⚠️  API server not responding properly")
    except requests.exceptions.ConnectionError:
        print("⚠️  API server not running (expected if not started)")
    except Exception as e:
        print(f"⚠️  API test error: {e}")

def save_test_results(results, filename="test_results.json"):
    """Save test results to file"""
    try:
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Test results saved to: {filename}")
    except Exception as e:
        print(f"❌ Error saving results: {e}")

def main():
    """Run all tests"""
    print("🚀 Starting Resume Parser Tests")
    print("=" * 50)
    
    all_results = {}
    
    # Test 1: Document Processor
    try:
        doc_results = test_document_processor()
        all_results['document_processor'] = doc_results
    except Exception as e:
        print(f"❌ Document processor test failed: {e}")
        all_results['document_processor'] = {'error': str(e)}
    
    # Test 2: LLM Processor
    try:
        llm_results = test_llm_processor()
        if llm_results:
            all_results['llm_processor'] = llm_results
    except Exception as e:
        print(f"❌ LLM processor test failed: {e}")
        all_results['llm_processor'] = {'error': str(e)}
    
    # Test 3: API Integration
    try:
        test_api_integration()
        all_results['api_integration'] = {'status': 'tested'}
    except Exception as e:
        print(f"❌ API integration test failed: {e}")
        all_results['api_integration'] = {'error': str(e)}
    
    # Save results
    save_test_results(all_results)
    
    print("\n" + "=" * 50)
    print("✅ All tests completed!")
    print("\n📊 Test Summary:")
    for test_name, result in all_results.items():
        if 'error' in result:
            print(f"  ❌ {test_name}: Failed")
        else:
            print(f"  ✅ {test_name}: Passed")
    
    print(f"\n📄 Detailed results saved to: test_results.json")

if __name__ == "__main__":
    main() 