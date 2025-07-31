#!/bin/bash

# Run tests with correct Python path
echo "🧪 Running tests ..."
python3 tests/test_pdf_resume_extraction.py

echo ""
echo "🧪 Running other tests..."
python3 tests/test_llm_client.py

echo ""
echo "✅ All tests completed!" 