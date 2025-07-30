import os
import tempfile
import shutil
from typing import List, Dict, Any
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from loguru import logger

from src.core.document_processor import ResumeDocumentProcessor
from src.core.llm_client import ResumeLLMProcessor
from src.core.config import app_config

# Initialize FastAPI app
app = FastAPI(
    title="Resume Parser API",
    description="Document parser using DocLing and vLLM",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize processors
document_processor = ResumeDocumentProcessor()
llm_processor = ResumeLLMProcessor()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Resume Parser API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "vllm_available": llm_processor.llm_client.health_check()
    }

@app.post("/parse")
async def parse_document(
    file: UploadFile = File(...),
    use_llm: bool = True,
    background_tasks: BackgroundTasks = None
):
    """Parse a single document"""
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in app_config.allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"File type {file_extension} not supported. Allowed: {app_config.allowed_extensions}"
            )
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
        
        try:
            # Process document
            logger.info(f"Processing document: {file.filename}")
            
            # Basic document processing
            result = document_processor.process_document(temp_file_path)
            
            if 'error' in result:
                raise HTTPException(status_code=500, detail=result['error'])
            
            # Enhance with LLM if requested and available
            if use_llm:
                logger.info("Enhancing extraction with LLM")
                result = llm_processor.enhance_extraction(result)
                
                # Add additional LLM-based extractions
                if result.get('raw_text'):
                    result['llm_summary'] = llm_processor.summarize_resume(result['raw_text'])
                    result['key_achievements'] = llm_processor.extract_key_achievements(result['raw_text'])
            
            # Add processing metadata
            result['processing_info'] = {
                'filename': file.filename,
                'file_size': file.size,
                'llm_enhanced': use_llm and llm_processor.llm_client.health_check(),
                'processing_time': 'completed'
            }
            
            return JSONResponse(content=result)
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/parse/batch")
async def parse_batch_documents(
    files: List[UploadFile] = File(...),
    use_llm: bool = True
):
    """Parse multiple documents in batch"""
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        results = []
        
        for file in files:
            try:
                # Validate file
                file_extension = Path(file.filename).suffix.lower()
                if file_extension not in app_config.allowed_extensions:
                    results.append({
                        'filename': file.filename,
                        'error': f"File type {file_extension} not supported"
                    })
                    continue
                
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                    shutil.copyfileobj(file.file, temp_file)
                    temp_file_path = temp_file.name
                
                try:
                    # Process document
                    result = document_processor.process_document(temp_file_path)
                    
                    if 'error' not in result and use_llm:
                        result = llm_processor.enhance_extraction(result)
                    
                    result['filename'] = file.filename
                    results.append(result)
                    
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
                        
            except Exception as e:
                results.append({
                    'filename': file.filename,
                    'error': str(e)
                })
        
        return JSONResponse(content={
            'total_files': len(files),
            'processed_files': len([r for r in results if 'error' not in r]),
            'failed_files': len([r for r in results if 'error' in r]),
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract/specific")
async def extract_specific_info(
    file: UploadFile = File(...),
    info_type: str = "skills"
):
    """Extract specific information from document"""
    try:
        # Validate info_type
        valid_types = ['skills', 'experience', 'education', 'contact']
        if info_type not in valid_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid info_type. Must be one of: {valid_types}"
            )
        
        # Extract text from document
        file_extension = Path(file.filename).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
        
        try:
            text = document_processor.extract_text(temp_file_path)
            if not text:
                raise HTTPException(status_code=400, detail="No text extracted from document")
            
            # Extract specific information using LLM
            extracted_info = llm_processor.extract_specific_info(text, info_type)
            
            return JSONResponse(content={
                'filename': file.filename,
                'info_type': info_type,
                'extracted_info': extracted_info
            })
            
        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        logger.error(f"Error extracting specific info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/vllm/status")
async def vllm_status():
    """Check vLLM server status"""
    is_healthy = llm_processor.llm_client.health_check()
    return {
        "vllm_available": is_healthy,
        "base_url": llm_processor.llm_client.base_url
    }

@app.post("/vllm/test")
async def test_vllm():
    """Test vLLM generation"""
    try:
        test_prompt = "Extract the name and email from this text: John Doe, john.doe@email.com"
        response = llm_processor.llm_client.generate(test_prompt, max_tokens=100)
        
        return {
            "success": True,
            "test_prompt": test_prompt,
            "response": response
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host=app_config.host,
        port=app_config.port,
        reload=app_config.debug
    ) 