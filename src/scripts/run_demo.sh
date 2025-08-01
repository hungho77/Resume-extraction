#!/bin/bash

# Run PDF inference demo
echo "🧪 Running PDF inference demo ..."

if [ $# -eq 0 ]; then
    echo "Usage: $0 <input> [-o <output>] [--batch]"
    echo ""
    echo "Examples:"
    echo "  Single file: $0 assets/resume.pdf"
    echo "  Single file with output: $0 assets/resume.pdf -o output/result.json"
    echo "  Directory: $0 data/INFORMATION-TECHNOLOGY"
    echo "  Directory with output: $0 data/INFORMATION-TECHNOLOGY -o output/INFORMATION-TECHNOLOGY"
    echo "  Batch mode: $0 assets/resume.pdf --batch"
    exit 1
fi

# Pass all arguments to the Python script
python3 src/demo/pdf_inference_demo.py "$@"

echo ""
echo "✅ Demo completed!" 