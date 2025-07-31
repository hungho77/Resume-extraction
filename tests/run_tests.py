#!/usr/bin/env python3
"""
Test Runner for Resume Parser
Runs all test files and provides a summary of results.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# Add parent directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def run_test_file(test_file: str) -> dict:
    """Run a single test file and return results"""
    print(f"\n🧪 Running {test_file}...")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        # Determine the working directory based on where the script is run from
        if os.path.exists(test_file):
            # Running from main directory
            cwd = os.getcwd()
        else:
            # Running from tests directory
            cwd = os.path.dirname(__file__)
            test_file = os.path.basename(test_file)
        
        # Run the test file
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=True,
            text=True,
            cwd=cwd
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ {test_file} completed successfully ({duration:.2f}s)")
            return {
                'file': test_file,
                'status': 'PASSED',
                'duration': duration,
                'output': result.stdout,
                'error': result.stderr
            }
        else:
            print(f"❌ {test_file} failed ({duration:.2f}s)")
            return {
                'file': test_file,
                'status': 'FAILED',
                'duration': duration,
                'output': result.stdout,
                'error': result.stderr
            }
            
    except Exception as e:
        print(f"❌ {test_file} crashed: {e}")
        return {
            'file': test_file,
            'status': 'CRASHED',
            'duration': 0,
            'output': '',
            'error': str(e)
        }

def run_all_tests():
    """Run all test files"""
    print("🚀 Resume Parser Test Suite")
    print("=" * 50)
    
    # Get all test files - handle both running from main dir and tests dir
    if os.path.exists('tests/test_example.py'):
        # Running from main directory
            test_files = [
        'tests/test_example.py',
        'tests/example_usage.py',
        'tests/test_qwen_resume.py',
        'tests/test_pdf_resume.py',
        'tests/test_llm_providers.py',
        'tests/test_llm_client.py'
    ]
    else:
        # Running from tests directory
        test_files = [
            'test_example.py',
            'example_usage.py', 
            'test_qwen_resume.py',
            'test_pdf_resume.py',
            'test_llm_providers.py'
        ]
    
    results = []
    total_passed = 0
    total_failed = 0
    
    for test_file in test_files:
        if os.path.exists(test_file):
            result = run_test_file(test_file)
            results.append(result)
            
            if result['status'] == 'PASSED':
                total_passed += 1
            else:
                total_failed += 1
        else:
            print(f"⚠️  {test_file} not found")
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 Test Summary")
    print("=" * 50)
    
    for result in results:
        status_icon = "✅" if result['status'] == 'PASSED' else "❌"
        print(f"{status_icon} {result['file']}: {result['status']} ({result['duration']:.2f}s)")
    
    print(f"\n📈 Overall Results:")
    print(f"   ✅ Passed: {total_passed}")
    print(f"   ❌ Failed: {total_failed}")
    print(f"   📊 Total: {len(results)}")
    
    if total_failed == 0:
        print("\n🎉 All tests passed!")
        return True
    else:
        print(f"\n⚠️  {total_failed} test(s) failed")
        return False

def run_specific_test(test_name: str):
    """Run a specific test file"""
    # Handle both running from main dir and tests dir
    if os.path.exists(f'tests/{test_name}.py'):
        test_file = f"tests/{test_name}.py"
    else:
        test_file = f"{test_name}.py"
    
    if not os.path.exists(test_file):
        print(f"❌ Test file {test_file} not found")
        return False
    
    result = run_test_file(test_file)
    
    if result['status'] == 'PASSED':
        print(f"\n✅ {test_file} passed!")
        return True
    else:
        print(f"\n❌ {test_file} failed!")
        if result['error']:
            print(f"Error: {result['error']}")
        return False

def show_help():
    """Show help information"""
    print("🚀 Resume Parser Test Runner")
    print("=" * 30)
    print("Usage:")
    print("  python run_tests.py              # Run all tests")
    print("  python run_tests.py test_example # Run specific test")
    print("  python run_tests.py help         # Show this help")
    print("\nAvailable tests:")
    print("  - test_example.py     # Basic functionality tests")
    print("  - example_usage.py    # Usage examples")
    print("  - test_qwen_resume.py # Qwen3-8B enhanced tests")

def main():
    """Main function"""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'help':
            show_help()
        else:
            # Run specific test
            run_specific_test(command)
    else:
        # Run all tests
        success = run_all_tests()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 