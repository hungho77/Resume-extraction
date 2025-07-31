# Resume Parser API Usage Examples

This directory contains comprehensive examples showing how to use the Resume Parser API with both Python and shell scripts.

## 📁 Files Overview

- **`api_usage_examples.py`** - Python examples with full API client
- **`api_usage_examples.sh`** - Shell script examples using curl
- **`start_servers.sh`** - Server management script
- **`README.md`** - This documentation

## 🚀 Quick Start

### 1. Start the Servers

```bash
# Start both vLLM and API servers
./examples/start_servers.sh start

# Check server status
./examples/start_servers.sh status

# Test API endpoints
./examples/start_servers.sh test
```

### 2. Run Python Examples

```bash
# Run all Python examples
python examples/api_usage_examples.py

# Or run individual examples
python -c "
from examples.api_usage_examples import ResumeParserAPI
api = ResumeParserAPI()
result = api.parse_single_resume('path/to/resume.pdf')
print(result)
"
```

### 3. Run Shell Script Examples

```bash
# Run all shell script examples
./examples/api_usage_examples.sh

# Make script executable first
chmod +x examples/api_usage_examples.sh
```

## 📋 API Endpoints

### Base URL
```
http://localhost:8080
```

### Available Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Check API health |
| `/parse` | POST | Parse single resume |
| `/parse/batch` | POST | Parse multiple resumes |
| `/extract/specific` | POST | Extract specific information |
| `/vllm/status` | GET | Check vLLM server status |

## 🐍 Python Examples

### Basic Usage

```python
from examples.api_usage_examples import ResumeParserAPI

# Initialize API client
api = ResumeParserAPI()

# Parse a single resume
result = api.parse_single_resume("resume.pdf")
print(result)
```

### Parse with LLM Enhancement

```python
# Parse with LLM enhancement (default)
result = api.parse_single_resume("resume.pdf", use_llm=True)

# Parse without LLM enhancement
result = api.parse_single_resume("resume.pdf", use_llm=False)
```

### Batch Processing

```python
# Process multiple resumes
resume_files = ["resume1.pdf", "resume2.docx", "resume3.txt"]
result = api.parse_batch_resumes(resume_files)
```

### Extract Specific Information

```python
# Extract specific information types
info_types = ["skills", "experience", "education", "contact"]

for info_type in info_types:
    result = api.extract_specific_info("resume.pdf", info_type)
    print(f"{info_type}: {result}")
```

### Error Handling

```python
# Check API health first
health = api.health_check()
if "error" in health:
    print(f"API not available: {health['error']}")
    return

# Parse with error handling
result = api.parse_single_resume("resume.pdf")
if "error" in result:
    print(f"Parse failed: {result['error']}")
else:
    print("Parse successful!")
```

## 🐚 Shell Script Examples

### Basic Resume Parsing

```bash
# Parse a single resume
curl -X POST "http://localhost:8080/parse" \
  -F "file=@resume.pdf" \
  -F "use_llm=true"
```

### Parse Without LLM

```bash
# Parse without LLM enhancement
curl -X POST "http://localhost:8080/parse" \
  -F "file=@resume.pdf" \
  -F "use_llm=false"
```

### Batch Processing

```bash
# Process multiple resumes
curl -X POST "http://localhost:8080/parse/batch" \
  -F "files=@resume1.pdf" \
  -F "files=@resume2.docx" \
  -F "files=@resume3.txt" \
  -F "use_llm=true"
```

### Extract Specific Information

```bash
# Extract skills
curl -X POST "http://localhost:8080/extract/specific" \
  -F "file=@resume.pdf" \
  -F "info_type=skills"

# Extract experience
curl -X POST "http://localhost:8080/extract/specific" \
  -F "file=@resume.pdf" \
  -F "info_type=experience"
```

### Health Checks

```bash
# Check API health
curl "http://localhost:8080/health"

# Check vLLM status
curl "http://localhost:8080/vllm/status"
```

## 🔧 Server Management

### Start Servers

```bash
# Start both servers
./examples/start_servers.sh start

# Start with custom ports
VLLM_PORT=8001 API_PORT=8081 ./examples/start_servers.sh start
```

### Check Status

```bash
# Check server status
./examples/start_servers.sh status

# Test API endpoints
./examples/start_servers.sh test
```

### Stop Servers

```bash
# Stop all servers
./examples/start_servers.sh stop

# Restart servers
./examples/start_servers.sh restart
```

### View Logs

```bash
# View server logs
./examples/start_servers.sh logs

# View specific log files
tail -f vllm.log    # vLLM server logs
tail -f api.log     # API server logs
```

## 📊 Response Format

### Successful Parse Response

```json
{
  "personal_info": {
    "name": "John Doe",
    "email": "john.doe@email.com",
    "phone": "(555) 123-4567",
    "location": "San Francisco, CA"
  },
  "education": [
    {
      "degree": "Bachelor of Science in Computer Science",
      "institution": "University of Technology",
      "graduation_date": "2018"
    }
  ],
  "experience": [
    {
      "title": "Senior Developer",
      "company": "Tech Corp",
      "duration": "2020-2023",
      "responsibilities": ["Led development of web applications", "Mentored junior developers"]
    }
  ],
  "skills": ["Python", "JavaScript", "React", "Node.js", "Docker", "AWS"],
  "summary": "Experienced software engineer with 5+ years in full-stack development.",
  "llm_summary": "John Doe is a senior software engineer with expertise in full-stack development...",
  "key_achievements": [
    "Led development of 3 major web applications",
    "Mentored 5 junior developers",
    "Improved system performance by 40%"
  ],
  "processing_info": {
    "filename": "resume.pdf",
    "processing_time": 2.5,
    "file_size": 245760
  }
}
```

### Error Response

```json
{
  "error": "File type .xyz not supported. Allowed: ['.pdf', '.docx', '.txt', '.doc']",
  "status_code": 400
}
```

## 🛠️ Advanced Usage

### Custom API Client

```python
import requests

class CustomResumeAPI:
    def __init__(self, base_url="http://localhost:8080"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def parse_resume(self, file_path, use_llm=True):
        with open(file_path, 'rb') as f:
            response = self.session.post(
                f"{self.base_url}/parse",
                files={'file': f},
                data={'use_llm': str(use_llm).lower()}
            )
        return response.json()

# Usage
api = CustomResumeAPI()
result = api.parse_resume("resume.pdf")
```

### Performance Testing

```bash
# Test response time
time curl -X POST "http://localhost:8080/parse" \
  -F "file=@resume.pdf" \
  -F "use_llm=true"

# Test with different file sizes
for file in small.pdf medium.pdf large.pdf; do
    echo "Testing $file..."
    time curl -X POST "http://localhost:8080/parse" \
      -F "file=@$file" \
      -F "use_llm=true"
done
```

### Monitoring

```bash
# Monitor API usage
watch -n 1 'curl -s "http://localhost:8080/health" | jq'

# Monitor server logs
tail -f api.log | grep "parse"

# Check GPU usage (if using GPU)
watch -n 1 nvidia-smi
```

## 🔍 Troubleshooting

### Common Issues

1. **API not responding**
   ```bash
   # Check if servers are running
   ./examples/start_servers.sh status
   
   # Check logs for errors
   ./examples/start_servers.sh logs
   ```

2. **vLLM server not available**
   ```bash
   # Check vLLM status
   curl "http://localhost:8080/vllm/status"
   
   # Restart vLLM server
   ./scripts/start_server.sh --cli
   ```

3. **File upload errors**
   ```bash
   # Check file format
   file resume.pdf
   
   # Check file size
   ls -lh resume.pdf
   ```

4. **Memory issues**
   ```bash
   # Check GPU memory
   nvidia-smi
   
   # Reduce batch size or use CPU
   export CUDA_VISIBLE_DEVICES=""
   ```

### Debug Mode

```bash
# Start API server in debug mode
python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8080 --log-level debug

# Start vLLM server with verbose output
./scripts/start_server.sh --cli --verbose
```

## 📝 Best Practices

### File Preparation

1. **Use supported formats**: PDF, DOCX, TXT
2. **Optimize file size**: Keep files under 10MB
3. **Ensure text quality**: Use clear, readable fonts
4. **Avoid scanned images**: Use native text when possible

### API Usage

1. **Check health first**: Always verify API is running
2. **Handle errors gracefully**: Implement proper error handling
3. **Use timeouts**: Set appropriate request timeouts
4. **Batch when possible**: Use batch processing for multiple files

### Performance

1. **Use LLM selectively**: Disable for simple extractions
2. **Monitor resources**: Watch GPU memory and CPU usage
3. **Cache results**: Store parsed results when appropriate
4. **Optimize requests**: Minimize unnecessary API calls

## 🤝 Contributing

To add new examples:

1. **Python examples**: Add to `api_usage_examples.py`
2. **Shell examples**: Add to `api_usage_examples.sh`
3. **Documentation**: Update this README
4. **Testing**: Test with different file types and sizes

## 📚 Additional Resources

- [Main README](../README.md) - Project overview and setup
- [API Documentation](../docs/api.md) - Detailed API reference
- [Configuration Guide](../docs/configuration.md) - Server configuration
- [Troubleshooting Guide](../docs/troubleshooting.md) - Common issues and solutions 