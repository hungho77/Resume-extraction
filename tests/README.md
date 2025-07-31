# Resume Parser Tests

This directory contains comprehensive tests for the Resume Parser system, including basic functionality tests, usage examples, and enhanced LLM processing tests.

## 📁 Test Files

### `test_example.py`
- **Purpose**: Basic functionality tests
- **Tests**: Document processing, LLM enhancement, API integration
- **Features**: 
  - Tests document processor with sample resume
  - Tests LLM processor with vLLM server
  - Tests API integration (when server is running)
  - Saves test results to JSON file

### `example_usage.py`
- **Purpose**: Usage examples and demonstrations
- **Tests**: Basic usage, LLM enhancement, API usage, batch processing
- **Features**:
  - Shows how to use the document processor
  - Demonstrates LLM enhancement capabilities
  - Shows API integration examples
  - Demonstrates batch processing

### `test_qwen_resume.py`
- **Purpose**: Advanced LLM processing tests with Qwen3-8B
- **Tests**: Enhanced resume processing with complex data
- **Features**:
  - Tests with complex resume data
  - Tests Qwen3-8B model capabilities
  - Tests specific information extraction
  - Tests API integration with enhanced processing

### `run_tests.py`
- **Purpose**: Test runner script
- **Features**:
  - Runs all tests with summary
  - Can run individual tests
  - Provides timing and status information
  - Works from both main directory and tests directory

## 🚀 Running Tests

### Run All Tests
```bash
# From main directory
python tests/run_tests.py

# From tests directory
cd tests
python run_tests.py
```

### Run Individual Tests
```bash
# Run specific test
python tests/run_tests.py test_example
python tests/run_tests.py example_usage
python tests/run_tests.py test_qwen_resume

# Or run directly
python tests/test_example.py
python tests/example_usage.py
python tests/test_qwen_resume.py
```

### Run Tests from Tests Directory
```bash
cd tests
python test_example.py
python example_usage.py
python test_qwen_resume.py
```

## 📊 Test Results

### Expected Output
```
🚀 Resume Parser Test Suite
==================================================

🧪 Running test_example.py...
==================================================
✅ test_example.py completed successfully (4.76s)

🧪 Running example_usage.py...
==================================================
✅ example_usage.py completed successfully (4.71s)

🧪 Running test_qwen_resume.py...
==================================================
✅ test_qwen_resume.py completed successfully (6.31s)

==================================================
📊 Test Summary
==================================================
✅ test_example.py: PASSED (4.76s)
✅ example_usage.py: PASSED (4.71s)
✅ test_qwen_resume.py: PASSED (6.31s)

📈 Overall Results:
   ✅ Passed: 3
   ❌ Failed: 0
   📊 Total: 3

🎉 All tests passed!
```

### Test Results Files
- `test_results.json` - Basic test results
- `qwen_test_results.json` - Qwen3-8B test results

## 🔧 Test Configuration

### Prerequisites
- Python 3.8+
- Resume Parser dependencies installed
- vLLM server running (for LLM tests)
- API server running (for API tests)

### Environment Setup
```bash
# Activate conda environment
conda activate resume_extraction

# Install dependencies
pip install -r requirements.txt

# Start vLLM server (optional, for LLM tests)
./scripts/start_server.sh --cli

# Start API server (optional, for API tests)
python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8080
```

## 📋 Test Coverage

### Document Processing
- ✅ PDF text extraction
- ✅ DOCX text extraction
- ✅ TXT file processing
- ✅ Personal information extraction
- ✅ Education information extraction
- ✅ Work experience extraction
- ✅ Skills extraction
- ✅ Contact information extraction

### LLM Enhancement
- ✅ vLLM server connectivity
- ✅ LLM-enhanced extraction
- ✅ Summary generation
- ✅ Key achievements extraction
- ✅ Specific information extraction

### API Integration
- ✅ Health check endpoints
- ✅ Document parsing endpoints
- ✅ Batch processing endpoints
- ✅ Specific extraction endpoints

### Error Handling
- ✅ Invalid file handling
- ✅ Missing file handling
- ✅ Server unavailability handling
- ✅ Network error handling

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Make sure you're in the correct directory
   cd /path/to/Resume-extraction
   
   # Check Python path
   python -c "import sys; print(sys.path)"
   ```

2. **vLLM Server Not Available**
   ```bash
   # Start vLLM server
   ./scripts/start_server.sh --cli
   
   # Check if it's running
   curl http://localhost:8000/health
   ```

3. **API Server Not Available**
   ```bash
   # Start API server
   python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8080
   
   # Check if it's running
   curl http://localhost:8080/health
   ```

4. **Memory Issues**
   ```bash
   # Check GPU memory
   nvidia-smi
   
   # Reduce batch size or use CPU
   export CUDA_VISIBLE_DEVICES=""
   ```

### Debug Mode
```bash
# Run tests with verbose output
python -v tests/test_example.py

# Run with debug logging
python tests/test_example.py 2>&1 | tee test_debug.log
```

## 📝 Adding New Tests

### Test Structure
```python
#!/usr/bin/env python3
"""
Test description
"""

import sys
import os

# Add parent directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.document_processor import ResumeDocumentProcessor

def test_function():
    """Test description"""
    print("🧪 Running test...")
    
    # Test code here
    
    print("✅ Test completed!")
    return True

if __name__ == "__main__":
    test_function()
```

### Best Practices
1. **Use descriptive test names**
2. **Include proper error handling**
3. **Clean up temporary files**
4. **Provide clear output messages**
5. **Save results to files when appropriate**

## 📚 Related Documentation

- [Main README](../README.md) - Project overview
- [API Examples](../examples/README.md) - API usage examples
- [Configuration Guide](../docs/configuration.md) - Server configuration
- [Troubleshooting Guide](../docs/troubleshooting.md) - Common issues 