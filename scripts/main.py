#!/usr/bin/env python3
"""
Main script for Resume Parser
Demonstrates usage of the document parser with DocLing and vLLM integration.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any

from src.core.document_processor import ResumeDocumentProcessor
from src.core.llm_client import ResumeLLMProcessor, VLLMClient
from src.core.config import vllm_config, app_config

def print_results(results: Dict[str, Any], filename: str):
    """Pretty print the parsing results"""
    print(f"\n{'='*60}")
    print(f"Results for: {filename}")
    print(f"{'='*60}")
    
    if 'error' in results:
        print(f"❌ Error: {results['error']}")
        return
    
    # Personal Information
    if 'personal_info' in results:
        print("\n📋 Personal Information:")
        for key, value in results['personal_info'].items():
            print(f"  {key.title()}: {value}")
    
    # Contact Information
    if 'contact' in results:
        print("\n📞 Contact Information:")
        for key, value in results['contact'].items():
            print(f"  {key.title()}: {value}")
    
    # Skills
    if 'skills' in results:
        print(f"\n💻 Skills ({len(results['skills'])} found):")
        for skill in results['skills']:
            print(f"  • {skill}")
    
    # Education
    if 'education' in results:
        print(f"\n🎓 Education ({len(results['education'])} entries):")
        for edu in results['education']:
            print(f"  • {edu.get('section', 'N/A')}")
    
    # Experience
    if 'experience' in results:
        print(f"\n💼 Work Experience ({len(results['experience'])} entries):")
        for exp in results['experience']:
            print(f"  • {exp.get('section', 'N/A')}")
    
    # LLM Summary
    if 'llm_summary' in results:
        print(f"\n🤖 LLM Summary:")
        print(f"  {results['llm_summary']}")
    
    # Key Achievements
    if 'key_achievements' in results:
        print(f"\n🏆 Key Achievements ({len(results['key_achievements'])} found):")
        for achievement in results['key_achievements']:
            print(f"  • {achievement}")
    
    # Metadata
    if 'metadata' in results:
        print(f"\n📊 Processing Metadata:")
        for key, value in results['metadata'].items():
            print(f"  {key.title()}: {value}")

def process_single_file(file_path: str, use_llm: bool = True) -> Dict[str, Any]:
    """Process a single document file"""
    print(f"Processing: {file_path}")
    
    # Initialize processors
    doc_processor = ResumeDocumentProcessor()
    llm_processor = ResumeLLMProcessor()
    
    # Check if vLLM is available
    if use_llm:
        vllm_available = llm_processor.llm_client.health_check()
        if not vllm_available:
            print("⚠️  vLLM server not available, proceeding without LLM enhancement")
            use_llm = False
    
    # Process document
    results = doc_processor.process_document(file_path)
    
    if 'error' in results:
        return results
    
    # Enhance with LLM if requested and available
    if use_llm:
        print("🔧 Enhancing with LLM...")
        results = llm_processor.enhance_extraction(results)
        
        # Add additional LLM-based extractions
        if results.get('raw_text'):
            results['llm_summary'] = llm_processor.summarize_resume(results['raw_text'])
            results['key_achievements'] = llm_processor.extract_key_achievements(results['raw_text'])
    
    return results

def process_directory(directory_path: str, use_llm: bool = True, output_file: str = None):
    """Process all documents in a directory"""
    directory = Path(directory_path)
    
    if not directory.exists():
        print(f"❌ Directory not found: {directory_path}")
        return
    
    # Find all supported files
    supported_extensions = app_config.allowed_extensions
    files = []
    
    for ext in supported_extensions:
        files.extend(directory.glob(f"*{ext}"))
        files.extend(directory.glob(f"*{ext.upper()}"))
    
    if not files:
        print(f"❌ No supported files found in {directory_path}")
        print(f"Supported extensions: {supported_extensions}")
        return
    
    print(f"📁 Found {len(files)} files to process")
    
    all_results = []
    
    for file_path in files:
        try:
            results = process_single_file(str(file_path), use_llm)
            results['filename'] = file_path.name
            all_results.append(results)
            
            # Print results
            print_results(results, file_path.name)
            
        except Exception as e:
            print(f"❌ Error processing {file_path.name}: {e}")
            all_results.append({
                'filename': file_path.name,
                'error': str(e)
            })
    
    # Save results to file if requested
    if output_file:
        try:
            with open(output_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\n💾 Results saved to: {output_file}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")

def check_vllm_status():
    """Check vLLM server status"""
    client = VLLMClient()
    is_healthy = client.health_check()
    
    print(f"🔍 vLLM Server Status:")
    print(f"  URL: {client.base_url}")
    print(f"  Status: {'✅ Available' if is_healthy else '❌ Not Available'}")
    
    if is_healthy:
        # Test generation
        try:
            test_response = client.generate("Hello, world!", max_tokens=10)
            print(f"  Test Response: {test_response}")
        except Exception as e:
            print(f"  Test Failed: {e}")
    
    return is_healthy

def main():
    parser = argparse.ArgumentParser(description="Resume Parser with DocLing and vLLM")
    parser.add_argument(
        "input",
        help="File or directory to process"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM enhancement"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file for results (JSON format)"
    )
    parser.add_argument(
        "--check-vllm",
        action="store_true",
        help="Check vLLM server status and exit"
    )
    parser.add_argument(
        "--test-vllm",
        action="store_true",
        help="Test vLLM generation and exit"
    )
    
    args = parser.parse_args()
    
    # Check vLLM status if requested
    if args.check_vllm:
        check_vllm_status()
        return
    
    # Test vLLM if requested
    if args.test_vllm:
        client = VLLMClient()
        if client.health_check():
            test_prompt = "Extract the name and email from this text: John Doe, john.doe@email.com"
            response = client.generate(test_prompt, max_tokens=100)
            print(f"Test Prompt: {test_prompt}")
            print(f"Response: {response}")
        else:
            print("❌ vLLM server not available")
        return
    
    # Process input
    input_path = Path(args.input)
    use_llm = not args.no_llm
    
    if not input_path.exists():
        print(f"❌ Input not found: {args.input}")
        return
    
    if input_path.is_file():
        # Process single file
        results = process_single_file(str(input_path), use_llm)
        print_results(results, input_path.name)
        
        # Save results if requested
        if args.output:
            try:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"\n💾 Results saved to: {args.output}")
            except Exception as e:
                print(f"❌ Error saving results: {e}")
    
    elif input_path.is_dir():
        # Process directory
        process_directory(str(input_path), use_llm, args.output)
    
    else:
        print(f"❌ Invalid input: {args.input}")

if __name__ == "__main__":
    main() 