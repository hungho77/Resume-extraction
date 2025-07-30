# Resume Parser with DocLing and Qwen3-8B

A powerful document parser that extracts structured information from resumes using DocLing for document processing and Qwen3-8B for LLM enhancement.

## 🏗️ Project Structure

```
Resume-extraction/
├── src/                    # Source code
│   ├── core/              # Core modules
│   │   ├── config.py      # Configuration management
│   │   ├── document_processor.py  # Document processing logic
│   │   └── llm_client.py  # LLM integration
│   ├── api/               # API related
│   │   └── server.py      # FastAPI server
│   └── utils/             # Utilities
├── scripts/               # Executable scripts
│   ├── vllm_server.py    # vLLM server script
│   ├── main.py           # CLI interface
│   └── start_servers.sh  # Server startup script
├── tests/                 # Test files
│   ├── test_qwen_resume.py
│   ├── test_example.py
│   └── example_usage.py
├── docs/                  # Documentation
│   ├── README.md         # Detailed documentation
│   └── SETUP.md          # Setup guide
└── requirements.txt
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start Servers
```bash
# Start both vLLM and API servers
bash scripts/start_servers.sh

# Or start only vLLM server
bash scripts/start_servers.sh --vllm-only

# Or start only API server
bash scripts/start_servers.sh --api-only
```

### 3. Test the System
```bash
# Test with Qwen3-8B
python tests/test_qwen_resume.py

# Use CLI interface
python scripts/main.py --help
```

## 🔧 Configuration

The system uses Qwen3-8B model with the following default settings:
- **Model**: `Qwen/Qwen3-8B`
- **Max Model Length**: 32768 tokens
- **GPU Memory Utilization**: 0.7
- **VRAM Requirement**: ~16GB

## 📋 Features

- **Document Processing**: Support for PDF, DOCX, and TXT files
- **OCR Integration**: DocLing with OCR for image-based PDFs
- **LLM Enhancement**: Qwen3-8B for advanced information extraction
- **REST API**: FastAPI-based web service
- **CLI Interface**: Command-line tool for batch processing

## 🛠️ Usage

### API Usage
```bash
# Parse a single document
curl -X POST "http://localhost:8080/parse" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@resume.pdf"

# Parse multiple documents
curl -X POST "http://localhost:8080/parse/batch" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@resume1.pdf" \
  -F "files=@resume2.pdf"
```

### CLI Usage
```bash
# Process a single file
python scripts/main.py resume.pdf

# Process a directory
python scripts/main.py /path/to/resumes/ --output results.json

# Check vLLM status
python scripts/main.py --check-vllm
```

## 📚 Documentation

- **Detailed Documentation**: See `docs/README.md`
- **Setup Guide**: See `docs/SETUP.md`
- **Examples**: See `tests/example_usage.py`

## 🔍 Testing

```bash
# Run comprehensive tests
python tests/test_qwen_resume.py

# Run basic tests
python tests/test_example.py

# See usage examples
python tests/example_usage.py
```

## 🐛 Troubleshooting

1. **vLLM Server Issues**: Check GPU memory with `nvidia-smi`
2. **Import Errors**: Ensure you're running from the project root
3. **Port Conflicts**: Use `--vllm-port` and `--api-port` flags

## 📄 License

This project is open source and available under the MIT License. 