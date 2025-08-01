# Resume Parser with DocLing and LLM

A comprehensive resume parsing system that combines **DocLing** for document parsing, **SmolDocLing** for OCR on scanned documents, and **LLM** for information extraction. Supports both **local VLM models** and **API key services** for flexible deployment.

## 🚀 Features

- **Multi-format Support**: PDF, DOCX, TXT files
- **OCR Capability**: Handles scanned documents using SmolDocLing VLM
- **Flexible LLM Integration**: Local models (vLLM) or cloud APIs (OpenAI, etc.)
- **Structured Extraction**: Name, email, phone, skills, education, experience, certifications
- **Evaluation Framework**: Comprehensive metrics and ground truth validation
- **API Server**: FastAPI-based REST API for production deployment

## 🏗️ Architecture

![Workflow Diagram](assets/workflow.png)

The workflow below illustrates the end-to-end resume extraction pipeline, from document ingestion (PDF, DOCX, scanned images) through OCR (if needed), document parsing, LLM-based information extraction, and outputting structured JSON results. This modular design enables flexible integration of local or cloud-based models and supports both batch and API-based processing.

## 🛠️ Tech Stack

## 📁 Project Structure

```
Resume-extraction/
├── data/
│   ├── GT/                           # Ground truth files (ID_gt.json)
│   ├── INFORMATION-TECHNOLOGY/       # Original PDF dataset
│   ├── Resume/                       # CSV data with resume information
│   └── SCAN-INFORMATION-TECHNOLOGY/  # Converted scanned PDFs
├── evaluate/
│   ├── run_evaluation.py            # Combined evaluation pipeline
│   ├── create_gt.py                 # Ground truth creation from CSV
│   ├── evaluate.py                  # Basic evaluation
│   ├── comprehensive_eval.py        # Comprehensive evaluation
│   └── summary.py                   # Evaluation summary
├── outputs/                         # JSON output files from processing
├── results/                         # Evaluation results and metrics
├── src/
│   ├── core/                       # Core processing modules
│   ├── api/                        # API server components
│   └── utils/                      # Utility functions
├── datasets/                       # Dataset processing scripts
├── scripts/                        # Shell scripts for automation
└── tests/                         # Test suites
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (for local VLM models)
- 16GB+ RAM (for large models)

### Setup

1. **Clone and Install Dependencies**:
```bash
git clone <repository-url>
cd Resume-extraction
pip install -r requirements.txt
```

2. **Environment Configuration**:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. **Choose LLM Configuration**:

**Option A: Local vLLM (Recommended for Production)**
```bash
# .env configuration
LLM_BASE_URL=http://localhost:8000/v1
LLM_MODEL=Qwen/Qwen3-8B
LLM_API_KEY=
DOCLING_USE_OCR=true
```

**Option B: OpenAI API (Easy Setup)**
```bash
# .env configuration
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
LLM_API_KEY=sk-your-key-here
DOCLING_USE_OCR=true
```

**Option C: Anthropic Claude (High Quality)**
```bash
# .env configuration
LLM_BASE_URL=https://api.anthropic.com/v1
LLM_MODEL=claude-3-5-sonnet-20241022
LLM_API_KEY=sk-ant-your-key-here
DOCLING_USE_OCR=true
```

## 🚀 Usage

### 1. Start LLM Server (Local Mode)

**For vLLM (Recommended)**:
```bash
# Start vLLM server
python src/api/vllm_server.py --model Qwen/Qwen3-8B --tensor-parallel-size 1

# Or use CLI
vllm serve Qwen/Qwen3-8B --host 0.0.0.0 --port 8000
```

### 2. Process Resumes

**Single File Processing**:
```bash
python scripts/run_demo.sh path/to/pdf_resume -o path/to/output
```

**Scrips Processing**:
```bash
python scripts/run_demo.sh path/to/resume_folder -o path/to/output_folder
```

**API Server**:
```bash
# Start API server
python src/api/resume_extraction_server.py

# Or use script
./scripts/start_resume_server.sh
```

### 3. Dataset Management

**Analyze PDF Dataset**:
```bash
cd datasets
python dataset.py
```

**Convert PDFs to Scanned Format**:
```bash
cd datasets
python convert_to_scanned.py
```

### 4. Evaluation Pipeline

**Combined Evaluation (Recommended)**:
```bash
cd evaluate

# Custom mode - direct evaluation with CSV data
python run_evaluation.py --csv-path ../data/Resume/Resume.csv --outputs-dir ../outputs

# Standard mode - uses existing evaluation scripts
python run_evaluation.py --mode standard

# With custom paths
python run_evaluation.py --csv-path /path/to/resumes.csv --outputs-dir /path/to/outputs --mode custom
```

**Individual Evaluation Steps**:
```bash
cd evaluate

# Create ground truth from CSV
python create_gt.py

# Basic evaluation
python evaluate.py

# Comprehensive evaluation
python comprehensive_eval.py

# Generate summary
python summary.py
```

## 📊 Data Structure

### Input Data Organization

```
data/
├── GT/                           # Ground truth files
│   ├── 16852973_gt.json        # Individual GT files by ID
│   ├── 10265057_gt.json
│   └── ...
├── INFORMATION-TECHNOLOGY/       # Original PDF dataset
│   ├── resume1.pdf
│   ├── resume2.pdf
│   └── ...
├── Resume/                       # CSV data
│   └── Resume.csv               # ID, Resume_str, Resume_html, Category
└── SCAN-INFORMATION-TECHNOLOGY/  # Converted scanned PDFs
    ├── resume1_scan.pdf
    ├── resume2_scan.pdf
    └── ...
```

### Output Structure

```
outputs/                          # Processing results
├── 16852973.json               # JSON output by ID
├── 10265057.json
└── ...

results/                          # Evaluation results
├── ground_truth.json           # Main ground truth
├── evaluation_results.json      # Basic evaluation
├── comprehensive_evaluation_results.json
├── custom_evaluation_results.json
└── results_index.json
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_BASE_URL` | LLM service URL | `http://localhost:8000/v1` |
| `LLM_MODEL` | Model name | `Qwen/Qwen3-8B` |
| `LLM_API_KEY` | API key (optional for vLLM) | - |
| `LLM_MAX_TOKENS` | Maximum tokens | `2048` |
| `LLM_TEMPERATURE` | Generation temperature | `0.1` |
| `DOCLING_USE_OCR` | Enable OCR processing | `true` |

### Model Configuration Examples

**Local vLLM Setup**:
```bash
# Install vLLM
pip install vllm

# Start server
python src/api/vllm_server.py --model Qwen/Qwen3-8B --tensor-parallel-size 1
```

**OpenAI Setup**:
```bash
# Get API key from OpenAI
export LLM_API_KEY=sk-your-key-here
export LLM_BASE_URL=https://api.openai.com/v1
export LLM_MODEL=gpt-4o-mini
```

## 📈 Evaluation Framework

### Metrics

- **Entity-Level Accuracy**: Exact/fuzzy matching for strings
- **List-Level Metrics**: Precision, recall, F1-score for lists
- **Overall Score**: Aggregated F1 across all fields
- **Performance Tiers**: Excellent (≥80%), Good (60-79%), Fair (40-59%), Poor (<40%)

### Evaluation Commands

```bash
# Quick evaluation
python run_evaluation.py --mode custom

# Full evaluation pipeline
python run_evaluation.py --mode standard

# Custom paths
python run_evaluation.py --csv-path data/Resume/Resume.csv --outputs-dir outputs
```

### Expected Performance

- **Text PDFs**: >85% accuracy target
- **Scanned PDFs**: >70% accuracy target (due to OCR errors)
- **Field-specific**: Name, email, phone should achieve >90% accuracy

## 🧪 Testing

```bash
# Run all tests
python tests/run_tests.py

# Run specific test
python tests/test_llm_client.py

# Test setup
python tests/test_setup.py
```

## 🚀 Deployment

### Production Setup

1. **Start vLLM Server**:
```bash
python src/api/vllm_server.py --model Qwen/Qwen3-8B --tensor-parallel-size 2
```

2. **Start API Server**:
```bash
python src/api/resume_extraction_server.py
```

3. **Monitor Performance**:
```bash
# Check server health
curl http://localhost:8001/health

# Process resume via API
curl -X POST http://localhost:8001/extract \
  -F "file=@resume.pdf" \
  -F "save_output=true"
```

### Docker Deployment

```bash
# Build image
docker build -t resume-parser .

# Run container
docker run -p 8001:8001 -p 8000:8000 resume-parser
```

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce `tensor_parallel_size`
   - Lower `gpu_memory_utilization`
   - Use smaller model

2. **vLLM Server Not Starting**:
   - Check CUDA installation
   - Verify model path
   - Check port availability

3. **Evaluation Errors**:
   - Ensure ground truth files exist in `data/GT/`
   - Verify CSV format matches expected structure
   - Check file permissions

4. **OCR Processing Issues**:
   - Install PyTorch with CUDA support
   - Check SmolDocLing model availability
   - Verify image processing libraries

### Performance Optimization

- **GPU Memory**: Use `--gpu-memory-utilization 0.8`
- **Batch Processing**: Process multiple files simultaneously
- **Model Selection**: Choose appropriate model size for your hardware
- **Caching**: Enable model caching for repeated requests

## 📚 API Documentation

### Endpoints

- `GET /`: Root endpoint with service info
- `GET /health`: Health check
- `POST /extract`: Extract entities from single resume
- `POST /extract/batch`: Process multiple resumes
- `POST /extract/specific`: Extract specific field
- `GET /status`: Detailed system status

### Example Usage

```bash
# Single file extraction
curl -X POST http://localhost:8001/extract \
  -F "file=@resume.pdf" \
  -F "save_output=true"

# Batch processing
curl -X POST http://localhost:8001/extract/batch \
  -F "files=@resume1.pdf" \
  -F "files=@resume2.pdf" \
  -F "save_outputs=true"
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **DocLing**: Document parsing and structure extraction
- **SmolDocLing**: Vision-language model for OCR
- **vLLM**: High-performance LLM serving
- **Qwen**: Base language model for information extraction 