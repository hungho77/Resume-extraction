#!/bin/bash

# Run PDF inference demo
echo "🧪 Running PDF inference demo ..."

if [ $# -eq 0 ]; then
    echo "Usage: $0 <pdf_file> [output_json]"
    echo "Example: $0 docs/examples_resume.pdf -o output/result.json"
    exit 1
fi

if [ $# -eq 2 ]; then
    # If two arguments provided, use -o flag
    python3 demo/pdf_inference_demo.py "$1" -o "$2"
else
    # Otherwise pass all arguments as-is
    python3 demo/pdf_inference_demo.py "$@"
fi

echo ""
echo "✅ Demo completed!" 