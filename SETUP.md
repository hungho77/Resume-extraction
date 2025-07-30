# Setup Guide - Resume Parser with DocLing and vLLM

This guide will help you set up and run the document parser system.

## Prerequisites

### System Requirements
- **Python 3.8+**
- **CUDA-compatible GPU** (recommended for vLLM)
- **8GB+ RAM** (16GB+ recommended)
- **Linux/Ubuntu** (tested on Ubuntu 20.04+)

### GPU Requirements (for vLLM)
- **NVIDIA GPU** with CUDA support
- **CUDA 11.8+** installed
- **8GB+ VRAM** (16GB+ recommended for larger models)

## Installation Steps

### 1. Clone and Setup
```bash
# Navigate to your workspace
cd /path/to/your/workspace

# Clone the repository (if using git)
# git clone <repository-url>
# cd ResumeParse

# Or if you already have the files, just navigate to the directory
cd ResumeParse
```

### 2. Install Dependencies
```bash
# Install Python dependencies
pip install -r requirements.txt

# If you encounter issues with torch, install it separately:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Verify Installation
```bash
# Test basic functionality
python test_example.py

# Check vLLM status
python main.py --check-vllm
```

## Quick Start

### Option 1: Using the Startup Script (Recommended)
```bash
# Start both servers in background
./start_servers.sh --background

# Or start with custom settings
./start_servers.sh --model "microsoft/DialoGPT-medium" --background
```

### Option 2: Manual Startup
```bash
# Terminal 1: Start vLLM server
python vllm_server.py

# Terminal 2: Start API server
python api_server.py
```

### Option 3: Command Line Only
```bash
# Process a single file
python main.py resume.pdf

# Process a directory
python main.py /path/to/resumes/

# Process without LLM enhancement
python main.py resume.pdf --no-llm
```

## Configuration

### Environment Variables
Create a `.env` file in the project root:
```bash
# vLLM Configuration
VLLM_MODEL=microsoft/DialoGPT-medium
VLLM_HOST=localhost
VLLM_PORT=8000

# API Configuration
APP_HOST=0.0.0.0
APP_PORT=8080
APP_DEBUG=true
```

### Model Selection
You can use different models with vLLM:

**Small Models (Faster, Less Memory)**
- `microsoft/DialoGPT-medium` (default)
- `microsoft/DialoGPT-small`
- `distilgpt2`

**Medium Models (Better Quality)**
- `gpt2-medium`
- `microsoft/DialoGPT-large`

**Large Models (Best Quality, More Memory)**
- `gpt2-large`
- `EleutherAI/gpt-neo-125M`

## Testing

### 1. Basic Test
```bash
python test_example.py
```

### 2. API Test
```bash
# Start servers first, then:
curl http://localhost:8080/health
```

### 3. Document Processing Test
```bash
# Create a test resume file
echo "John Doe, Software Engineer, john@email.com" > test_resume.txt

# Process it
python main.py test_resume.txt
```

## Troubleshooting

### Common Issues

#### 1. CUDA/GPU Issues
```bash
# Check CUDA installation
nvidia-smi

# Check PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available())"

# If CUDA not available, install:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 2. Memory Issues
```bash
# Reduce GPU memory usage in config.py
gpu_memory_utilization: float = 0.5  # Change from 0.9 to 0.5

# Or use a smaller model
./start_servers.sh --model "microsoft/DialoGPT-small"
```

#### 3. Port Conflicts
```bash
# Check what's using the ports
lsof -i :8000
lsof -i :8080

# Kill processes if needed
kill -9 <PID>

# Or use different ports
./start_servers.sh --vllm-port 8001 --api-port 8081
```

#### 4. Import Errors
```bash
# Reinstall dependencies
pip uninstall -r requirements.txt
pip install -r requirements.txt

# Or install missing packages individually
pip install docling vllm fastapi uvicorn
```

### Debug Mode
```bash
# Enable debug logging
export APP_DEBUG=true
python api_server.py

# Check logs for detailed error messages
```

## Performance Optimization

### 1. GPU Memory
- Use smaller models for limited VRAM
- Adjust `gpu_memory_utilization` in config
- Consider CPU-only mode for testing

### 2. Batch Processing
- Use batch endpoints for multiple files
- Process files in parallel when possible

### 3. Model Selection
- Start with smaller models for testing
- Upgrade to larger models for production

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
Description=Resume Parser API
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/ResumeParse
ExecStart=/usr/bin/python3 api_server.py
Restart=always

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

### Common Commands Reference
```bash
# Start everything
./start_servers.sh --background

# Check status
python main.py --check-vllm

# Test processing
python test_example.py

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

Happy parsing! 🚀 