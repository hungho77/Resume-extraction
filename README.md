# Resume Parser with DocLing and Qwen3-30B-A3B-Instruct-2507-FP8

A comprehensive document parser that combines DocLing for document processing with OCR capabilities and Qwen3-30B-A3B-Instruct-2507-FP8 model hosted by vLLM for enhanced information extraction from resumes and other documents.

## Features

- **Multi-format Support**: PDF, DOCX, TXT files with DocLing OCR extraction
- **DocLing Integration**: Advanced document processing with OCR capabilities
- **Qwen3-30B-A3B-Instruct-2507-FP8 Integration**: State-of-the-art 30.5B parameter model with FP8 quantization
- **REST API**: FastAPI-based web service for easy integration
- **Batch Processing**: Process multiple documents at once
- **Structured Output**: JSON format with extracted information
- **Configurable**: Easy configuration through environment variables
- **OCR-Enhanced PDF Processing**: DocLing with OCR for better text extraction from complex layouts

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Document      │    │   DocLing       │    │   vLLM Server   │
│   Input         │───▶│   Processor     │───▶│   (Qwen3-30B)   │
│   (PDF/DOCX)    │    │   + OCR         │    │   + FP8         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Structured    │    │   Enhanced      │
                       │   Data          │    │   Extraction    │
                       └─────────────────┘    └─────────────────┘
```

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd ResumeParse
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up environment variables** (optional):
```bash
export VLLM_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
export VLLM_HOST="localhost"
export VLLM_PORT=8000
export APP_HOST="0.0.0.0"
export APP_PORT=8080
```

## Quick Start

### 1. Start vLLM Server with Qwen3-30B-A3B-Instruct-2507-FP8

First, start the vLLM server to host Qwen3-30B-A3B-Instruct-2507-FP8 model:

```bash
python vllm_server.py
```

Or with custom model:
```bash
python vllm_server.py --model "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8" --port 8000
```

**Note**: Qwen3-30B-A3B-Instruct-2507-FP8 requires significant GPU memory (24GB+ VRAM recommended)

### 2. Start API Server

In another terminal, start the FastAPI server:

```bash
python api_server.py
```

The API will be available at `http://localhost:8080`

### 3. Process Documents

#### Using the Command Line Tool

Process a single file:
```bash
python main.py resume.pdf
```

Process a directory:
```bash
python main.py /path/to/resumes/
```

Process without LLM enhancement:
```bash
python main.py resume.pdf --no-llm
```

Save results to JSON:
```bash
python main.py resume.pdf -o results.json
```

#### Using the API

Upload and parse a document:
```bash
curl -X POST "http://localhost:8080/parse" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@resume.pdf"
```

Parse multiple documents:
```bash
curl -X POST "http://localhost:8080/parse/batch" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@resume1.pdf" \
  -F "files=@resume2.pdf"
```

Extract specific information:
```bash
curl -X POST "http://localhost:8080/extract/specific" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@resume.pdf" \
  -F "info_type=skills"
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_MODEL` | `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` | Model to use with vLLM |
| `VLLM_HOST` | `localhost` | vLLM server host |
| `VLLM_PORT` | `8000` | vLLM server port |
| `APP_HOST` | `0.0.0.0` | API server host |
| `APP_PORT` | `8080` | API server port |
| `APP_DEBUG` | `true` | Enable debug mode |

### Supported File Formats

- **PDF**: `.pdf` (enhanced with DocLing OCR)
- **Word Documents**: `.docx`, `.doc`
- **Text Files**: `.txt`

## API Endpoints

### Health Check
```
GET /health
```

### Parse Single Document
```
POST /parse
Parameters:
- file: UploadFile (required)
- use_llm: bool (default: true)
```

### Parse Multiple Documents
```
POST /parse/batch
Parameters:
- files: List[UploadFile] (required)
- use_llm: bool (default: true)
```

### Extract Specific Information
```
POST /extract/specific
Parameters:
- file: UploadFile (required)
- info_type: str (default: "skills")
  Options: "skills", "experience", "education", "contact"
```

### vLLM Status
```
GET /vllm/status
```

### Test vLLM
```
POST /vllm/test
```

## Output Format

The parser extracts the following structured information:

```json
{
  "personal_info": {
    "name": "John Doe",
    "email": "john.doe@email.com",
    "phone": "123-456-7890"
  },
  "contact": {
    "email": "john.doe@email.com",
    "phone": "123-456-7890",
    "linkedin": "linkedin.com/in/johndoe",
    "github": "github.com/johndoe"
  },
  "skills": [
    "python",
    "javascript",
    "react",
    "aws",
    "docker",
    "kubernetes"
  ],
  "education": [
    {
      "section": "Bachelor of Science in Computer Science",
      "keyword": "education"
    }
  ],
  "experience": [
    {
      "section": "Software Engineer at Tech Corp (2020-2023)",
      "keyword": "experience"
    }
  ],
  "summary": "Professional summary...",
  "llm_summary": "AI-generated summary using Qwen3-30B-A3B-Instruct-2507-FP8...",
  "key_achievements": [
    "Led team of 5 developers",
    "Improved performance by 40%"
  ],
  "metadata": {
    "file_path": "/path/to/resume.pdf",
    "file_size": 12345,
    "text_length": 1500,
    "chunks": 3,
    "processing_method": "docling_with_ocr",
    "ocr_enabled": true,
    "ocr_language": "eng"
  }
}
```

## Advanced Usage

### Custom Model Configuration

Edit `config.py` to customize the vLLM model:

```python
@dataclass
class VLLMConfig:
    model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"  # Qwen3-30B model
    host: str = "localhost"
    port: int = 8000
    tensor_parallel_size: int = 1
    max_model_len: int = 32768  # Increased for Qwen3-30B
    gpu_memory_utilization: float = 0.7  # Adjusted for Qwen3-30B
```

### Custom Document Processing

Extend the `ResumeDocumentProcessor` class to add custom extraction logic:

```python
class CustomDocumentProcessor(ResumeDocumentProcessor):
    def extract_custom_info(self, text: str) -> Dict[str, Any]:
        # Add your custom extraction logic
        pass
```

### Integration with Other Services

The API can be easily integrated with other services:

```python
import requests

# Upload and parse document
with open('resume.pdf', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8080/parse', files=files)
    results = response.json()
```

## Performance Optimization

### GPU Memory Management

For Qwen3-30B-A3B-Instruct-2507-FP8 model:
- **Minimum**: 24GB VRAM
- **Recommended**: 32GB+ VRAM
- **CPU Fallback**: Available but slower
- **FP8 Quantization**: Reduces memory usage while maintaining quality

### Memory Optimization

```bash
# Reduce GPU memory usage
export VLLM_GPU_MEMORY_UTILIZATION=0.6

# Use smaller model for testing
python vllm_server.py --model "Qwen/Qwen2.5-7B-Instruct"
```

## Troubleshooting

### vLLM Server Issues

1. **Check if vLLM is running**:
```bash
python main.py --check-vllm
```

2. **Test vLLM generation**:
```bash
python main.py --test-vllm
```

3. **Common vLLM issues**:
   - Ensure you have enough GPU memory (24GB+ for Qwen3-30B-A3B-Instruct-2507-FP8)
   - Check if the model is compatible with vLLM
   - Verify CUDA installation
   - Monitor GPU usage: `nvidia-smi`

### API Server Issues

1. **Check API health**:
```bash
curl http://localhost:8080/health
```

2. **Check logs** for detailed error messages

3. **Common API issues**:
   - Port conflicts (change port in config)
   - File size limits (adjust in config)
   - CORS issues (check middleware settings)

### Document Processing Issues

1. **Unsupported file format**: Check `allowed_extensions` in config
2. **Text extraction failed**: DocLing OCR may need additional setup
3. **LLM enhancement failed**: Check vLLM server status

## Performance Tips

1. **GPU Memory**: Use appropriate `gpu_memory_utilization` for your GPU
2. **Batch Processing**: Use batch endpoints for multiple documents
3. **Model Selection**: Start with smaller models for testing
4. **Caching**: Implement caching for repeated requests
5. **Monitor Resources**: Use `nvidia-smi` to monitor GPU usage
6. **FP8 Benefits**: The model uses FP8 quantization for better performance

## Production Deployment

### 1. Docker (Recommended)
```dockerfile
# Create Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8080

CMD ["python", "api_server.py"]
```

### 2. Systemd Service
```bash
# Create service file
sudo nano /etc/systemd/system/resume-parser.service

[Unit]
Description=Resume Parser API with Qwen3-30B-A3B-Instruct-2507-FP8
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/ResumeParse
ExecStart=/usr/bin/python3 api_server.py
Restart=always
Environment=VLLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507-FP8

[Install]
WantedBy=multi-user.target
```

### 3. Nginx Reverse Proxy
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Monitoring

### 1. Health Checks
```bash
# API health
curl http://localhost:8080/health

# vLLM health
curl http://localhost:8000/health
```

### 2. Logs
```bash
# Check API logs
tail -f api_server.log

# Check vLLM logs
tail -f vllm_server.log
```

### 3. Metrics
- Monitor GPU memory usage: `nvidia-smi`
- Monitor API response times
- Monitor file processing success rates

## Support

### Getting Help
1. Check the troubleshooting section above
2. Review the logs for error messages
3. Test with a simple file first
4. Verify all dependencies are installed
5. Check GPU memory availability

### Common Commands Reference
```bash
# Start everything
./start_servers.sh --background

# Check status
python main.py --check-vllm

# Test processing
python test_qwen_resume.py

# Process files
python main.py resume.pdf -o results.json

# Stop servers
pkill -f "python.*server.py"
```

## Next Steps

1. **Test with your own resumes**
2. **Customize the extraction logic** in `document_processor.py`
3. **Add your own models** to the vLLM configuration
4. **Integrate with your existing systems** using the API
5. **Scale up** for production use

## Acknowledgments

- Based on the Grok solution for resume parsing
- Uses DocLing for document processing with OCR capabilities
- Uses Qwen3-30B-A3B-Instruct-2507-FP8 model hosted by vLLM for enhanced extraction
- Enhanced with FP8 quantization for better performance
- Built with FastAPI for the web service 