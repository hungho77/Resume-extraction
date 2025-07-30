#!/usr/bin/env python3
"""
Example Usage of Resume Parser
Simple demonstration of how to use the document parser.
"""

import json
import tempfile
import os

def create_simple_resume():
    """Create a simple resume for demonstration"""
    return """
Alice Johnson
Data Scientist
alice.johnson@email.com | (555) 987-6543

EXPERIENCE
Data Scientist at TechCorp (2021-2023)
- Built machine learning models for customer segmentation
- Improved prediction accuracy by 25%
- Used Python, scikit-learn, pandas, numpy

Data Analyst at StartupXYZ (2019-2021)
- Analyzed user behavior data
- Created dashboards with Tableau
- Used SQL, Python, Excel

EDUCATION
Master of Science in Data Science
University of Data | 2017-2019

Bachelor of Science in Mathematics
State University | 2013-2017

SKILLS
Python, R, SQL, Machine Learning, Statistics, Tableau, AWS
"""

def example_basic_usage():
    """Example of basic usage without LLM"""
    print("📄 Example 1: Basic Document Processing")
    print("-" * 40)
    
    from src.core.document_processor import ResumeDocumentProcessor
    
    # Create a temporary file with sample resume
    sample_text = create_simple_resume()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(sample_text)
        temp_file = f.name
    
    try:
        # Initialize processor
        processor = ResumeDocumentProcessor()
        
        # Process document
        results = processor.process_document(temp_file)
        
        # Display results
        print("✅ Processing completed!")
        print(f"Name: {results.get('personal_info', {}).get('name', 'N/A')}")
        print(f"Email: {results.get('contact', {}).get('email', 'N/A')}")
        print(f"Skills: {', '.join(results.get('skills', []))}")
        
    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.unlink(temp_file)

def example_with_llm():
    """Example of usage with LLM enhancement"""
    print("\n🤖 Example 2: Processing with LLM Enhancement")
    print("-" * 40)
    
    from src.core.document_processor import ResumeDocumentProcessor
    from src.core.llm_client import ResumeLLMProcessor
    
    # Create a temporary file with sample resume
    sample_text = create_simple_resume()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(sample_text)
        temp_file = f.name
    
    try:
        # Initialize processors
        doc_processor = ResumeDocumentProcessor()
        llm_processor = ResumeLLMProcessor()
        
        # Check if vLLM is available
        if not llm_processor.llm_client.health_check():
            print("⚠️  vLLM server not available, using basic processing only")
            results = doc_processor.process_document(temp_file)
        else:
            # Process with LLM enhancement
            print("🔧 Using LLM enhancement...")
            results = doc_processor.process_document(temp_file)
            results = llm_processor.enhance_extraction(results)
            
            # Add LLM-based extractions
            if results.get('raw_text'):
                results['llm_summary'] = llm_processor.summarize_resume(results['raw_text'])
                results['key_achievements'] = llm_processor.extract_key_achievements(results['raw_text'])
        
        # Display enhanced results
        print("✅ Enhanced processing completed!")
        print(f"Name: {results.get('personal_info', {}).get('name', 'N/A')}")
        print(f"Email: {results.get('contact', {}).get('email', 'N/A')}")
        print(f"Skills: {', '.join(results.get('skills', []))}")
        
        if 'llm_summary' in results:
            print(f"AI Summary: {results['llm_summary'][:100]}...")
        
        if 'key_achievements' in results:
            print(f"Key Achievements: {len(results['key_achievements'])} found")
        
    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.unlink(temp_file)

def example_api_usage():
    """Example of using the API"""
    print("\n🌐 Example 3: Using the API")
    print("-" * 40)
    
    import requests
    
    # Create a temporary file with sample resume
    sample_text = create_simple_resume()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(sample_text)
        temp_file = f.name
    
    try:
        # Check if API is running
        try:
            response = requests.get("http://localhost:8080/health", timeout=5)
            if response.status_code == 200:
                print("✅ API server is running")
                
                # Upload and parse document
                with open(temp_file, 'rb') as f:
                    files = {'file': f}
                    response = requests.post(
                        "http://localhost:8080/parse",
                        files=files,
                        params={'use_llm': 'true'}
                    )
                
                if response.status_code == 200:
                    results = response.json()
                    print("✅ API processing completed!")
                    print(f"Name: {results.get('personal_info', {}).get('name', 'N/A')}")
                    print(f"Email: {results.get('contact', {}).get('email', 'N/A')}")
                    print(f"Skills: {', '.join(results.get('skills', []))}")
                else:
                    print(f"❌ API error: {response.status_code} - {response.text}")
            else:
                print("⚠️  API server not responding properly")
        except requests.exceptions.ConnectionError:
            print("⚠️  API server not running (start with: python api_server.py)")
        except Exception as e:
            print(f"⚠️  API test error: {e}")
    
    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.unlink(temp_file)

def example_batch_processing():
    """Example of batch processing"""
    print("\n📁 Example 4: Batch Processing")
    print("-" * 40)
    
    from document_processor import ResumeDocumentProcessor
    
    # Create multiple sample resumes
    resumes = [
        ("Alice", create_simple_resume()),
        ("Bob", """
Bob Smith
Software Engineer
bob.smith@email.com | (555) 111-2222

EXPERIENCE
Software Engineer at BigTech (2020-2023)
- Developed web applications using React and Node.js
- Led team of 3 developers
- Used JavaScript, Python, AWS

EDUCATION
Bachelor of Science in Computer Science
Tech University | 2016-2020

SKILLS
JavaScript, Python, React, Node.js, AWS, Git
"""),
    ]
    
    # Create temporary files
    temp_files = []
    try:
        for name, content in resumes:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(content)
                temp_files.append((name, f.name))
        
        # Process all files
        processor = ResumeDocumentProcessor()
        all_results = []
        
        for name, file_path in temp_files:
            print(f"Processing {name}'s resume...")
            results = processor.process_document(file_path)
            results['name'] = name
            all_results.append(results)
        
        # Display batch results
        print("✅ Batch processing completed!")
        for result in all_results:
            name = result.get('name', 'Unknown')
            skills = ', '.join(result.get('skills', []))
            print(f"  {name}: {skills}")
        
    finally:
        # Clean up
        for name, file_path in temp_files:
            if os.path.exists(file_path):
                os.unlink(file_path)

def main():
    """Run all examples"""
    print("🚀 Resume Parser Examples")
    print("=" * 50)
    
    # Example 1: Basic usage
    example_basic_usage()
    
    # Example 2: With LLM
    example_with_llm()
    
    # Example 3: API usage
    example_api_usage()
    
    # Example 4: Batch processing
    example_batch_processing()
    
    print("\n" + "=" * 50)
    print("✅ All examples completed!")
    print("\n💡 Tips:")
    print("  - Start vLLM server: python vllm_server.py")
    print("  - Start API server: python api_server.py")
    print("  - Use both: ./start_servers.sh")
    print("  - Test everything: python test_example.py")

if __name__ == "__main__":
    main() 