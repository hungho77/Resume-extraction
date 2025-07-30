#!/usr/bin/env python3
"""
Test script for Qwen3-8B Enhanced Resume Parser
Demonstrates the improved capabilities with Qwen3-8B model.
"""

import json
import tempfile
import os
from pathlib import Path

from src.core.document_processor import ResumeDocumentProcessor
from src.core.llm_client import ResumeLLMProcessor, VLLMClient

def create_complex_resume():
    """Create a complex resume text for testing Qwen3-8B capabilities"""
    complex_resume = """
ALEXANDER CHEN
Senior Software Engineer & Machine Learning Specialist
alex.chen@techcorp.com | (555) 123-4567 | linkedin.com/in/alexchen | github.com/alexchen

PROFESSIONAL SUMMARY
Innovative software engineer with 8+ years of experience in full-stack development, 
machine learning, and cloud architecture. Proven track record of leading cross-functional 
teams and delivering scalable solutions that impact millions of users. Expert in Python, 
JavaScript, and cloud technologies with a passion for AI/ML applications.

TECHNICAL SKILLS
Programming Languages: Python, JavaScript, TypeScript, Java, C++, SQL, R, Go
Frameworks & Libraries: React, Angular, Vue.js, Django, Flask, Spring Boot, TensorFlow, PyTorch
Cloud & DevOps: AWS (EC2, S3, Lambda, ECS), Azure, GCP, Docker, Kubernetes, Jenkins, Git
Databases: PostgreSQL, MongoDB, Redis, Elasticsearch, Cassandra, DynamoDB
Machine Learning: TensorFlow, PyTorch, Scikit-learn, Pandas, NumPy, Jupyter, MLflow
Tools & Platforms: VS Code, IntelliJ, Jira, Confluence, Slack, Figma, Tableau

WORK EXPERIENCE

Senior Software Engineer | TechCorp Inc. | 2021-Present
• Led development of microservices architecture serving 5M+ users with 99.9% uptime
• Implemented CI/CD pipeline reducing deployment time by 70% and deployment frequency by 5x
• Mentored 8 junior developers and conducted 200+ code reviews, improving code quality by 40%
• Technologies: Python, Django, React, AWS, Docker, Kubernetes, PostgreSQL, Redis
• Achievements: Reduced API response time by 60%, improved system reliability by 85%

Machine Learning Engineer | AI Startup | 2019-2021
• Developed recommendation engine using TensorFlow and PyTorch, improving user engagement by 35%
• Built real-time data processing pipeline using Apache Kafka and Spark, handling 1M+ events/day
• Collaborated with data scientists to deploy ML models with 95% accuracy
• Technologies: Python, TensorFlow, PyTorch, Apache Spark, Kafka, AWS SageMaker, MongoDB
• Achievements: Increased recommendation accuracy by 25%, reduced model training time by 50%

Software Engineer | BigTech Corp | 2017-2019
• Developed RESTful APIs and frontend applications using modern JavaScript frameworks
• Improved application performance by 45% through optimization and caching strategies
• Participated in agile development processes and contributed to open-source projects
• Technologies: JavaScript, Node.js, React, MongoDB, Redis, AWS, Git
• Achievements: Reduced page load time by 40%, improved user satisfaction scores by 30%

EDUCATION
Master of Science in Computer Science | Stanford University | 2015-2017 | GPA: 3.9/4.0
• Specialization: Artificial Intelligence and Machine Learning
• Thesis: "Deep Learning Approaches for Natural Language Processing"

Bachelor of Science in Computer Science | UC Berkeley | 2011-2015 | GPA: 3.8/4.0
• Minor: Mathematics
• Dean's List: All semesters

CERTIFICATIONS
• AWS Certified Solutions Architect - Professional
• Google Cloud Professional Machine Learning Engineer
• Certified Kubernetes Administrator (CKA)
• Certified Scrum Master (CSM)

PROJECTS
AI-Powered Resume Parser | Full-stack application with ML integration
• Built with React, Python, TensorFlow, and AWS
• Processes 10,000+ resumes with 95% accuracy
• GitHub: github.com/alexchen/resume-parser

Real-time Chat Application | Scalable messaging platform
• Developed with Node.js, React, Socket.io, and MongoDB
• Handles 100,000+ concurrent users
• GitHub: github.com/alexchen/chat-app

Machine Learning Pipeline | Automated ML model training and deployment
• Built with Python, TensorFlow, Docker, and Kubernetes
• Reduces model deployment time by 80%
• GitHub: github.com/alexchen/ml-pipeline

LANGUAGES
English (Native), Mandarin Chinese (Fluent), Spanish (Conversational)

INTERESTS
Open Source Contribution, Machine Learning Research, Cloud Architecture, 
System Design, Technical Writing, Mentoring
"""
    return complex_resume

def test_qwen_enhanced_processing():
    """Test the enhanced processing with Qwen3-8B"""
    print("🚀 Testing Qwen3-8B Enhanced Resume Parser")
    print("=" * 60)
    
    # Create sample resume file
    sample_text = create_complex_resume()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(sample_text)
        temp_file = f.name
    
    try:
        # Initialize processors
        doc_processor = ResumeDocumentProcessor()
        llm_processor = ResumeLLMProcessor()
        
        print("📄 Step 1: Basic Document Processing...")
        results = doc_processor.process_document(temp_file)
        
        if 'error' in results:
            print(f"❌ Error in basic processing: {results['error']}")
            return
        
        print("✅ Basic processing completed!")
        print(f"  📋 Personal Info: {results.get('personal_info', {})}")
        print(f"  📞 Contact Info: {results.get('contact', {})}")
        print(f"  💻 Skills Found: {len(results.get('skills', []))}")
        print(f"  🎓 Education Entries: {len(results.get('education', []))}")
        print(f"  💼 Experience Entries: {len(results.get('experience', []))}")
        
        # Check vLLM availability
        print("\n🤖 Step 2: Checking Qwen3-8B Availability...")
        if not llm_processor.llm_client.health_check():
            print("⚠️  Qwen3-8B server not available, skipping LLM enhancement")
            print("   Start vLLM server with: python vllm_server.py")
            return
        
        print("✅ Qwen3-8B server is available!")
        
        # Test LLM enhancement
        print("\n🔧 Step 3: Testing Qwen3-8B Enhancement...")
        enhanced_results = llm_processor.enhance_extraction(results)
        
        print("✅ LLM enhancement completed!")
        
        # Test specific extractions
        print("\n📝 Step 4: Testing Specific Extractions...")
        
        # Test summary generation
        summary = llm_processor.summarize_resume(results.get('raw_text', ''))
        print(f"  📝 AI Summary: {summary[:150]}...")
        
        # Test achievements extraction
        achievements = llm_processor.extract_key_achievements(results.get('raw_text', ''))
        print(f"  🏆 Key Achievements: {len(achievements)} found")
        for i, achievement in enumerate(achievements[:3], 1):
            print(f"    {i}. {achievement}")
        
        # Test specific info extraction
        skills_info = llm_processor.extract_specific_info(results.get('raw_text', ''), 'skills')
        print(f"  💻 Enhanced Skills: {skills_info[:100]}...")
        
        # Save results
        output_file = "qwen_test_results.json"
        with open(output_file, 'w') as f:
            json.dump(enhanced_results, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
        
        # Print final summary
        print("\n" + "=" * 60)
        print("✅ Qwen3-8B Enhanced Processing Test Completed!")
        print("\n📊 Summary:")
        print(f"  📄 File processed: {temp_file}")
        print(f"  🤖 Model used: Qwen3-8B")
        print(f"  📋 Personal info extracted: {bool(enhanced_results.get('personal_info'))}")
        print(f"  💻 Skills detected: {len(enhanced_results.get('skills', []))}")
        print(f"  🎓 Education entries: {len(enhanced_results.get('education', []))}")
        print(f"  💼 Experience entries: {len(enhanced_results.get('experience', []))}")
        print(f"  📝 AI summary generated: {bool(enhanced_results.get('llm_summary'))}")
        print(f"  🏆 Achievements extracted: {len(enhanced_results.get('key_achievements', []))}")
        
    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.unlink(temp_file)

def test_api_integration():
    """Test API integration with Qwen3-8B"""
    print("\n🌐 Testing API Integration with Qwen3-8B...")
    
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
                
                # Test Qwen3-8B generation
                test_response = requests.post("http://localhost:8080/vllm/test", timeout=10)
                if test_response.status_code == 200:
                    test_result = test_response.json()
                    print(f"✅ Qwen3-8B Test: {test_result}")
                else:
                    print("⚠️  Qwen3-8B test failed")
            else:
                print("⚠️  Could not check vLLM status")
        else:
            print("⚠️  API server not responding properly")
    except requests.exceptions.ConnectionError:
        print("⚠️  API server not running (start with: python api_server.py)")
    except Exception as e:
        print(f"⚠️  API test error: {e}")

def main():
    """Run all Qwen3-8B tests"""
    print("🚀 Qwen3-8B Enhanced Resume Parser Test Suite")
    print("=" * 60)
    
    # Test 1: Enhanced processing
    test_qwen_enhanced_processing()
    
    # Test 2: API integration
    test_api_integration()
    
    print("\n" + "=" * 60)
    print("✅ All Qwen3-8B tests completed!")
    print("\n💡 Next Steps:")
    print("  - Test with your own resume files")
    print("  - Try different document formats (PDF, DOCX)")
    print("  - Monitor GPU usage with 'nvidia-smi'")
    print("  - Check the generated JSON results")

if __name__ == "__main__":
    main() 