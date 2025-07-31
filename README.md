# Resume Parser

A high-performance resume parsing system that extracts structured information from resumes using advanced document processing (DocLing) and Large Language Models (vLLM with Qwen3).

## 🚀 Features

- **📄 Multi-Format Support**: Parse PDF, DOCX, and TXT resume files
- **🔍 Advanced OCR**: Extract text from scanned documents and images using DocLing and SmolDocLing
- **🤖 Unified LLM Client**: Single client for vLLM and OpenAI with just different base URLs
- **⚡ High Performance**: Multi-GPU support with vLLM for fast processing
- **📊 Structured Output**: Extract personal info, education, experience, skills, and more
- **🌐 REST API**: Easy-to-use FastAPI endpoints for integration
- **🔄 Batch Processing**: Process multiple resumes simultaneously
- **⚙️ Environment Configuration**: Easy setup with .env file

## 📋 Extracted Information

The system extracts the following structured data from resumes:

- **Personal Information**: Name, email, phone, location
- **Education**: Degrees, institutions, graduation dates
- **Work Experience**: Job titles, companies, dates, responsibilities
- **Skills**: Technical skills, soft skills, certifications
- **Summary**: Professional summary and objectives
- **Key Achievements**: Notable accomplishments and metrics
- **Contact Information**: Email, phone, LinkedIn, portfolio links

## 🛠️ Prerequisites

- **Python 3.8+**
- **CUDA-compatible GPU(s)** with at least 8GB VRAM (recommended)
- **vLLM 0.8.5+** for LLM inference
- **PyTorch** with CUDA support

## 📦 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Resume-extraction
```

### 2. Install Dependencies

```bash
# Install all requirements
pip install -r requirements.txt

# Or install the package
pip install -e .
```

### 3. Verify Installation

```bash
python scripts/test_setup.py
```

## 🚀 Quick Start

### 1. Configure Environment

```bash
# Copy the example environment file
cp env.example .env

# Edit .env file with your configuration
# For vLLM: Set LLM_BASE_URL to your vLLM server (no API key needed)
# For OpenAI: Set LLM_BASE_URL to OpenAI API and LLM_API_KEY to your API key
```

### 2. Start the vLLM Server (Optional)

```bash
# Start with default settings (2 GPUs)
./scripts/start_server.sh --cli

# Start with single GPU
./scripts/start_server.sh --gpus 1 --cli

# Check GPU setup first
./scripts/start_server.sh --check-gpu
```

### 3. Start the Resume Parser API

```bash
# Start the FastAPI server
python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```

### 4. Parse Your First Resume

```bash
# Using curl
curl -X POST "http://localhost:8000/parse" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your/resume.pdf"

# Using Python
import requests

with open('resume.pdf', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/parse',
        files={'file': f}
    )
    result = response.json()
    print(result)
```

## 📚 Usage Examples

### Basic Resume Parsing

```python
from src.core.document_processor import ResumeDocumentProcessor

# Initialize processor
processor = ResumeDocumentProcessor()

# Parse a resume
result = processor.process_document("path/to/resume.pdf")

# Access extracted information
print(f"Name: {result['personal_info']['name']}")
print(f"Email: {result['personal_info']['email']}")
print(f"Skills: {result['skills']}")
```

### Using the LLM Enhancement

```python
from src.core.llm_client import ResumeLLMProcessor

# Initialize LLM processor
llm_processor = ResumeLLMProcessor()

# Enhance extraction with LLM
enhanced_result = llm_processor.enhance_extraction(result)

# Get LLM-generated summary
summary = llm_processor.summarize_resume(result['raw_text'])

# Extract key achievements
achievements = llm_processor.extract_key_achievements(result['raw_text'])
```

### Batch Processing

```python
# Process multiple resumes
resumes = ["resume1.pdf", "resume2.docx", "resume3.txt"]
results = []

for resume in resumes:
    result = processor.process_document(resume)
    results.append(result)
```

## 🌐 API Endpoints

### Parse Single Resume

```http
POST /parse
Content-Type: multipart/form-data

Parameters:
- file: Resume file (PDF, DOCX, TXT)
- use_llm: Boolean (default: true)

Response:
{
  "personal_info": {...},
  "education": [...],
  "experience": [...],
  "skills": [...],
  "summary": "...",
  "llm_summary": "...",
  "key_achievements": [...]
}
```

### Batch Processing

```http
POST /parse/batch
Content-Type: multipart/form-data

Parameters:
- files: Multiple resume files
- use_llm: Boolean (default: true)

Response:
{
  "results": [...],
  "summary": {...}
}
```

### Extract Specific Information

```http
POST /extract/specific
Content-Type: multipart/form-data

Parameters:
- file: Resume file
- info_type: "skills", "experience", "education", etc.

Response:
{
  "extracted_info": [...]
}
```

### Health Check

```http
GET /health

Response:
{
  "status": "healthy",
  "vllm_available": true
}
```

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `Qwen/Qwen3-8B` | LLM model to use |
| `TENSOR_PARALLEL_SIZE` | `2` | Number of GPUs |
| `MAX_MODEL_LEN` | `32768` | Maximum sequence length |
| `GPU_MEMORY_UTILIZATION` | `0.8` | GPU memory usage |
| `HOST` | `0.0.0.0` | API server host |
| `PORT` | `8000` | API server port |

### Supported File Formats

- **PDF**: Native text and OCR for scanned documents
- **DOCX**: Microsoft Word documents
- **TXT**: Plain text files

## 🔧 Advanced Usage

### Custom GPU Configuration

```bash
# Use specific GPU devices
./scripts/start_server.sh --gpu-devices '1,0' --cli

# Adjust memory utilization
./scripts/start_server.sh --gpu-memory-utilization 0.7 --cli

# Use different model
./scripts/start_server.sh --model Qwen/Qwen3-30B-A3B-Instruct-2507 --cli
```

### Production Deployment

```bash
# Run in background
nohup ./scripts/start_server.sh --cli > vllm.log 2>&1 &
nohup uvicorn src.api.server:app --host 0.0.0.0 --port 8000 > api.log 2>&1 &

# With reverse proxy (nginx)
# Configure nginx to proxy requests to localhost:8000
```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 📊 Performance Optimization

### GPU Memory Management

- **Single GPU**: Use `--gpus 1` for smaller models
- **Multi-GPU**: Use `--gpus 2` or more for better throughput
- **Memory Utilization**: Adjust based on available VRAM

### Batch Size Optimization

The system automatically adjusts batch sizes:
- **Single GPU**: `max_num_batched_tokens=4096`
- **Multi-GPU**: `max_num_batched_tokens=8192`

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce GPU memory utilization
   ./scripts/start_server.sh --gpu-memory-utilization 0.5 --cli
   ```

2. **vLLM Not Found**
   ```bash
   pip install vllm
   python scripts/test_setup.py
   ```

3. **Document Processing Errors**
   ```bash
   # Check file format support
   # Ensure file is not corrupted
   # Verify OCR dependencies (tesseract)
   ```

4. **API Connection Issues**
   ```bash
   # Check if servers are running
   curl http://localhost:8000/health
   curl http://localhost:8000/vllm/status
   ```

## 📁 Project Structure

```
Resume-extraction/
├── src/
│   ├── core/
│   │   ├── document_processor.py  # Document processing with DocLing
│   │   ├── llm_client.py         # LLM integration with vLLM
│   │   └── config.py             # Configuration management
│   ├── api/
│   │   └── server.py             # FastAPI REST server
│   └── utils/                    # Utility functions
├── scripts/
│   ├── start_server.sh           # vLLM server startup
│   ├── main.py                   # CLI interface
│   └── test_setup.py            # Installation verification
├── tests/                        # Test suite
├── docs/                         # Documentation
└── requirements.txt              # Dependencies
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

[Add your license information here]

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review GPU memory requirements
3. Verify vLLM installation
4. Check system logs for detailed error messages

## 🔗 Related Projects

- [DocLing](https://github.com/docling-ai/docling) - Document processing library
- [vLLM](https://github.com/vllm-project/vllm) - High-performance LLM inference
- [Qwen](https://github.com/QwenLM/Qwen) - Large language models 