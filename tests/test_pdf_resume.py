#!/usr/bin/env python3
"""
Test PDF Resume Processing
Tests the resume parser with a real PDF file from docs/examples_resume.pdf
"""

import sys
import os
import json
import time
from pathlib import Path

# Add parent directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.document_processor import ResumeDocumentProcessor
from src.core.llm_client import ResumeLLMProcessor

def test_pdf_resume_processing():
    """Test processing the PDF resume file"""
    print("🚀 PDF Resume Processing Test")
    print("=" * 50)
    
    # Path to the PDF resume file
    pdf_path = "docs/examples_resume.pdf"
    
    # Check if file exists
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return False
    
    print(f"📄 Processing PDF file: {pdf_path}")
    print(f"📊 File size: {os.path.getsize(pdf_path) / 1024:.1f} KB")
    
    try:
        # Initialize processor
        processor = ResumeDocumentProcessor()
        
        # Process the PDF
        print("\n🔄 Processing PDF resume...")
        start_time = time.time()
        
        results = processor.process_document(pdf_path)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"✅ PDF processing completed in {processing_time:.2f} seconds")
        
        # Display results
        print("\n📋 Extracted Information:")
        print("-" * 30)
        
        # Personal Information
        if 'personal_info' in results:
            personal = results['personal_info']
            print(f"👤 Name: {personal.get('name', 'N/A')}")
            print(f"📧 Email: {personal.get('email', 'N/A')}")
            print(f"📞 Phone: {personal.get('phone', 'N/A')}")
        
        # Contact Information
        if 'contact' in results:
            contact = results['contact']
            print(f"📧 Contact Email: {contact.get('email', 'N/A')}")
            print(f"📞 Contact Phone: {contact.get('phone', 'N/A')}")
            print(f"🔗 LinkedIn: {contact.get('linkedin', 'N/A')}")
        
        # Skills
        if 'skills' in results:
            skills = results['skills']
            print(f"💻 Skills ({len(skills)}): {', '.join(skills[:10])}{'...' if len(skills) > 10 else ''}")
        
        # Education
        if 'education' in results:
            education = results['education']
            print(f"🎓 Education ({len(education)} entries):")
            for i, edu in enumerate(education[:3]):  # Show first 3
                print(f"   {i+1}. {edu.get('degree', 'N/A')} from {edu.get('institution', 'N/A')}")
        
        # Experience
        if 'experience' in results:
            experience = results['experience']
            print(f"💼 Experience ({len(experience)} entries):")
            for i, exp in enumerate(experience[:3]):  # Show first 3
                print(f"   {i+1}. {exp.get('title', 'N/A')} at {exp.get('company', 'N/A')}")
        
        # Summary
        if 'summary' in results:
            summary = results['summary']
            print(f"📝 Summary: {summary[:100]}{'...' if len(summary) > 100 else ''}")
        
        # Raw text length
        if 'raw_text' in results:
            raw_text = results['raw_text']
            print(f"📄 Raw text length: {len(raw_text)} characters")
        
        return results
        
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        return False

def test_pdf_with_llm_enhancement():
    """Test PDF processing with LLM enhancement"""
    print("\n🤖 PDF Processing with LLM Enhancement")
    print("=" * 50)
    
    pdf_path = "docs/examples_resume.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return False
    
    try:
        # Initialize processors
        doc_processor = ResumeDocumentProcessor()
        llm_processor = ResumeLLMProcessor()
        
        # Check if vLLM is available
        if not llm_processor.llm_client.health_check():
            print("⚠️  vLLM server not available, skipping LLM enhancement")
            return False
        
        print("🔄 Processing PDF with LLM enhancement...")
        start_time = time.time()
        
        # Process document
        results = doc_processor.process_document(pdf_path)
        
        # Enhance with LLM
        enhanced_results = llm_processor.enhance_extraction(results)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"✅ Enhanced processing completed in {processing_time:.2f} seconds")
        
        # Display enhanced results
        print("\n📋 Enhanced Results:")
        print("-" * 30)
        
        # LLM Summary
        if 'llm_summary' in enhanced_results:
            llm_summary = enhanced_results['llm_summary']
            print(f"🤖 LLM Summary: {llm_summary[:150]}{'...' if len(llm_summary) > 150 else ''}")
        
        # Key Achievements
        if 'key_achievements' in enhanced_results:
            achievements = enhanced_results['key_achievements']
            print(f"🏆 Key Achievements ({len(achievements)}):")
            for i, achievement in enumerate(achievements[:3]):  # Show first 3
                print(f"   {i+1}. {achievement[:80]}{'...' if len(achievement) > 80 else ''}")
        
        return enhanced_results
        
    except Exception as e:
        print(f"❌ Error in LLM enhancement: {e}")
        return False

def test_pdf_specific_extractions():
    """Test specific information extraction from PDF"""
    print("\n🔍 PDF Specific Information Extraction")
    print("=" * 50)
    
    pdf_path = "docs/examples_resume.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return False
    
    try:
        # Initialize processors
        doc_processor = ResumeDocumentProcessor()
        llm_processor = ResumeLLMProcessor()
        
        # Extract text first
        text = doc_processor.extract_text(pdf_path)
        print(f"📄 Extracted text length: {len(text)} characters")
        
        # Test specific extractions
        extractions = {}
        
        # Skills extraction
        print("\n💻 Extracting skills...")
        skills_info = llm_processor.extract_specific_info(text, 'skills')
        extractions['skills'] = skills_info
        print(f"   Found: {skills_info}")
        
        # Experience extraction
        print("\n💼 Extracting experience...")
        experience_info = llm_processor.extract_specific_info(text, 'experience')
        extractions['experience'] = experience_info
        print(f"   Found: {experience_info}")
        
        # Education extraction
        print("\n🎓 Extracting education...")
        education_info = llm_processor.extract_specific_info(text, 'education')
        extractions['education'] = education_info
        print(f"   Found: {education_info}")
        
        # Contact extraction
        print("\n📞 Extracting contact...")
        contact_info = llm_processor.extract_specific_info(text, 'contact')
        extractions['contact'] = contact_info
        print(f"   Found: {contact_info}")
        
        return extractions
        
    except Exception as e:
        print(f"❌ Error in specific extractions: {e}")
        return False

def test_pdf_performance():
    """Test PDF processing performance"""
    print("\n⏱️  PDF Processing Performance Test")
    print("=" * 50)
    
    pdf_path = "docs/examples_resume.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return False
    
    try:
        processor = ResumeDocumentProcessor()
        
        # Test multiple runs for performance
        times = []
        results = []
        
        print("🔄 Running performance test (3 iterations)...")
        
        for i in range(3):
            print(f"   Run {i+1}/3...")
            start_time = time.time()
            
            result = processor.process_document(pdf_path)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            times.append(processing_time)
            results.append(result)
            
            print(f"   ✅ Run {i+1} completed in {processing_time:.2f}s")
        
        # Calculate statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"\n📊 Performance Results:")
        print(f"   ⏱️  Average time: {avg_time:.2f}s")
        print(f"   🏃 Fastest time: {min_time:.2f}s")
        print(f"   🐌 Slowest time: {max_time:.2f}s")
        print(f"   📈 Time variance: {max_time - min_time:.2f}s")
        
        return {
            'times': times,
            'average': avg_time,
            'min': min_time,
            'max': max_time,
            'results': results
        }
        
    except Exception as e:
        print(f"❌ Error in performance test: {e}")
        return False

def save_test_results(results, filename="pdf_test_results.json"):
    """Save test results to JSON file"""
    try:
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"💾 Results saved to: {filename}")
        return True
    except Exception as e:
        print(f"❌ Error saving results: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 PDF Resume Processing Test Suite")
    print("=" * 60)
    
    all_results = {}
    
    # Test 1: Basic PDF processing
    try:
        print("\n📋 Test 1: Basic PDF Processing")
        basic_results = test_pdf_resume_processing()
        if basic_results:
            all_results['basic_processing'] = basic_results
            print("✅ Basic PDF processing test passed")
        else:
            print("❌ Basic PDF processing test failed")
            all_results['basic_processing'] = {'error': 'Failed'}
    except Exception as e:
        print(f"❌ Basic PDF processing test crashed: {e}")
        all_results['basic_processing'] = {'error': str(e)}
    
    # Test 2: LLM enhancement
    try:
        print("\n📋 Test 2: LLM Enhancement")
        llm_results = test_pdf_with_llm_enhancement()
        if llm_results:
            all_results['llm_enhancement'] = llm_results
            print("✅ LLM enhancement test passed")
        else:
            print("❌ LLM enhancement test failed")
            all_results['llm_enhancement'] = {'error': 'Failed'}
    except Exception as e:
        print(f"❌ LLM enhancement test crashed: {e}")
        all_results['llm_enhancement'] = {'error': str(e)}
    
    # Test 3: Specific extractions
    try:
        print("\n📋 Test 3: Specific Extractions")
        extraction_results = test_pdf_specific_extractions()
        if extraction_results:
            all_results['specific_extractions'] = extraction_results
            print("✅ Specific extractions test passed")
        else:
            print("❌ Specific extractions test failed")
            all_results['specific_extractions'] = {'error': 'Failed'}
    except Exception as e:
        print(f"❌ Specific extractions test crashed: {e}")
        all_results['specific_extractions'] = {'error': str(e)}
    
    # Test 4: Performance test
    try:
        print("\n📋 Test 4: Performance Test")
        performance_results = test_pdf_performance()
        if performance_results:
            all_results['performance'] = performance_results
            print("✅ Performance test passed")
        else:
            print("❌ Performance test failed")
            all_results['performance'] = {'error': 'Failed'}
    except Exception as e:
        print(f"❌ Performance test crashed: {e}")
        all_results['performance'] = {'error': str(e)}
    
    # Save results
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    passed_tests = 0
    total_tests = len(all_results)
    
    for test_name, result in all_results.items():
        if 'error' not in result:
            print(f"✅ {test_name}: PASSED")
            passed_tests += 1
        else:
            print(f"❌ {test_name}: FAILED - {result['error']}")
    
    print(f"\n📈 Overall Results:")
    print(f"   ✅ Passed: {passed_tests}")
    print(f"   ❌ Failed: {total_tests - passed_tests}")
    print(f"   📊 Total: {total_tests}")
    
    # Save detailed results
    save_test_results(all_results, "pdf_test_results.json")
    
    if passed_tests == total_tests:
        print("\n🎉 All PDF tests passed!")
        return True
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 