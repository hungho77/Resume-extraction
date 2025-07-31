#!/usr/bin/env python3
"""
Resume Extraction Server
Reimplemented with pipeline approach similar to PDF inference demo
"""

import os
import tempfile
import shutil
import json
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from loguru import logger

from src.core.document_processor import ResumeDocumentProcessor

# Initialize FastAPI app
app = FastAPI(
    title="Resume Extraction API",
    description="Resume parser using DocLing and LLM pipeline",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize document processor
document_processor = ResumeDocumentProcessor()


def process_resume_pipeline(
    file_path: str, save_output: bool = False, output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process resume using the pipeline approach

    Args:
        file_path: Path to input file
        save_output: Whether to save output to JSON file
        output_path: Path to output JSON file (optional)

    Returns:
        Dictionary containing extracted entities and metadata
    """
    logger.info("🧪 Resume Processing Pipeline")
    logger.info("=" * 50)

    # Check if file exists
    resume_file = Path(file_path)
    if not resume_file.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info(f"📄 Input file: {resume_file}")
    logger.info(f"📄 File size: {resume_file.stat().st_size} bytes")

    try:
        # Process the resume
        logger.info("\n📝 Processing resume...")
        result = document_processor.extract_entities_from_resume(str(resume_file))

        if "error" in result:
            logger.error(f"❌ Processing failed: {result['error']}")
            return result

        # Extract the required fields
        entities = result.get("entities", {})
        validation = result.get("validation", {})
        metadata = result.get("metadata", {})

        # Create structured output with required fields
        structured_output = {
            "name": entities.get("name", ""),
            "email": entities.get("email", ""),
            "phone": entities.get("phone", ""),
            "skills": entities.get("skills", []),
            "education": entities.get("education", []),
            "experience": entities.get("experience", []),
            "certifications": entities.get("certifications", []),
            "languages": entities.get("languages", []),
            "metadata": {
                "file_path": metadata.get("file_path", str(resume_file)),
                "file_size": metadata.get("file_size", resume_file.stat().st_size),
                "text_length": metadata.get("text_length", 0),
                "processing_method": metadata.get("processing_method", "unknown"),
                "is_markdown_format": metadata.get("is_markdown_format", False),
                "completeness_score": validation.get("completeness_score", 0.0),
                "is_valid": validation.get("is_valid", False),
                "errors": validation.get("errors", []),
                "warnings": validation.get("warnings", []),
            },
        }

        # Log results summary
        logger.info("\n✅ Processing completed successfully!")
        logger.info(f"📊 File processed: {metadata.get('file_path', 'unknown')}")
        logger.info(f"📊 File size: {metadata.get('file_size', 0)} bytes")
        logger.info(f"📊 Text length: {metadata.get('text_length', 0)} characters")
        logger.info(
            f"📊 Processing method: {metadata.get('processing_method', 'unknown')}"
        )
        logger.info(
            f"📊 Is markdown format: {metadata.get('is_markdown_format', False)}"
        )
        logger.info(
            f"📊 Completeness score: {validation.get('completeness_score', 0):.2f}"
        )
        logger.info(f"📊 Is valid: {validation.get('is_valid', False)}")
        logger.info(f"📊 Errors: {len(validation.get('errors', []))}")
        logger.info(f"📊 Warnings: {len(validation.get('warnings', []))}")

        # Log extracted fields
        logger.info("\n📋 Extracted Information:")
        logger.info(f"  👤 Name: {structured_output['name']}")
        logger.info(f"  📧 Email: {structured_output['email']}")
        logger.info(f"  📞 Phone: {structured_output['phone']}")
        logger.info(f"  💻 Skills: {len(structured_output['skills'])} items")
        logger.info(f"  🎓 Education: {len(structured_output['education'])} entries")
        logger.info(f"  💼 Experience: {len(structured_output['experience'])} entries")
        logger.info(
            f"  🏆 Certifications: {len(structured_output['certifications'])} items"
        )
        logger.info(f"  🌍 Languages: {len(structured_output['languages'])} items")

        # Save to JSON file if requested
        if save_output and output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(structured_output, f, indent=2, ensure_ascii=False)

            logger.info(f"\n💾 Results saved to: {output_file}")
            logger.info(f"📄 Output file size: {output_file.stat().st_size} bytes")

        return structured_output

    except Exception as e:
        error_result = {
            "error": str(e),
            "metadata": {
                "file_path": str(resume_file),
                "file_size": resume_file.stat().st_size if resume_file.exists() else 0,
            },
        }
        logger.error(f"❌ Processing failed: {e}")
        return error_result


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Resume Extraction API",
        "version": "1.0.0",
        "status": "running",
        "pipeline": "DocLing + LLM",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test document processor
        processor_healthy = True
        try:
            # Simple test to check if processor can be initialized
            _ = ResumeDocumentProcessor()
        except Exception as e:
            processor_healthy = False
            logger.error(f"Document processor health check failed: {e}")

        return {
            "status": "healthy",
            "document_processor": processor_healthy,
            "timestamp": time.time(),
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e), "timestamp": time.time()}


@app.post("/extract")
async def extract_resume(
    file: UploadFile = File(...),
    save_output: bool = Form(False),
    output_filename: Optional[str] = Form(None),
):
    """
    Extract structured information from resume using pipeline approach

    Args:
        file: Uploaded resume file
        save_output: Whether to save output to JSON file
        output_filename: Custom output filename (optional)
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in [".pdf", ".txt", ".docx"]:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_extension} not supported. Allowed: .pdf, .txt, .docx",
            )

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=file_extension
        ) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name

        try:
            # Determine output path if saving is requested
            output_path = None
            if save_output:
                if output_filename:
                    output_path = f"output/{output_filename}"
                else:
                    output_path = f"output/{Path(file.filename).stem}.json"

            # Process resume using pipeline
            result = process_resume_pipeline(temp_file_path, save_output, output_path)

            if "error" in result:
                raise HTTPException(status_code=500, detail=result["error"])

            # Add API-specific metadata
            result["api_metadata"] = {
                "filename": file.filename,
                "file_size": file.size,
                "processing_time": time.time(),
                "endpoint": "/extract",
            }

            return JSONResponse(content=result)

        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    except Exception as e:
        logger.error(f"Error processing resume: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract/batch")
async def extract_batch_resumes(
    files: List[UploadFile] = File(...),
    save_outputs: bool = Form(False),
    output_directory: Optional[str] = Form("output"),
):
    """
    Extract structured information from multiple resumes using pipeline approach

    Args:
        files: List of uploaded resume files
        save_outputs: Whether to save outputs to JSON files
        output_directory: Directory to save JSON files (optional)
    """
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")

        results = {
            "total_files": len(files),
            "successful_files": 0,
            "failed_files": 0,
            "results": [],
            "summary": {
                "total_entities_found": 0,
                "average_completeness": 0.0,
                "total_errors": 0,
                "total_warnings": 0,
            },
        }

        for i, file in enumerate(files, 1):
            try:
                # Validate file
                file_extension = Path(file.filename).suffix.lower()
                if file_extension not in [".pdf", ".txt", ".docx"]:
                    results["results"].append(
                        {
                            "filename": file.filename,
                            "error": f"File type {file_extension} not supported",
                        }
                    )
                    results["failed_files"] += 1
                    continue

                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=file_extension
                ) as temp_file:
                    shutil.copyfileobj(file.file, temp_file)
                    temp_file_path = temp_file.name

                try:
                    # Determine output path if saving is requested
                    output_path = None
                    if save_outputs:
                        output_path = (
                            f"{output_directory}/{Path(file.filename).stem}.json"
                        )

                    # Process resume using pipeline
                    result = process_resume_pipeline(
                        temp_file_path, save_outputs, output_path
                    )

                    if "error" not in result:
                        results["successful_files"] += 1

                        # Update summary statistics
                        metadata = result.get("metadata", {})
                        results["summary"]["total_entities_found"] += sum(
                            [
                                len(result.get("skills", [])),
                                len(result.get("education", [])),
                                len(result.get("experience", [])),
                                len(result.get("certifications", [])),
                                len(result.get("languages", [])),
                            ]
                        )

                        if metadata.get("completeness_score"):
                            results["summary"]["average_completeness"] += metadata[
                                "completeness_score"
                            ]

                        results["summary"]["total_errors"] += len(
                            metadata.get("errors", [])
                        )
                        results["summary"]["total_warnings"] += len(
                            metadata.get("warnings", [])
                        )

                    else:
                        results["failed_files"] += 1

                    result["filename"] = file.filename
                    results["results"].append(result)

                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)

            except Exception as e:
                results["failed_files"] += 1
                results["results"].append({"filename": file.filename, "error": str(e)})

        # Calculate final statistics
        if results["successful_files"] > 0:
            results["summary"]["average_completeness"] /= results["successful_files"]

        # Add API-specific metadata
        results["api_metadata"] = {
            "processing_time": time.time(),
            "endpoint": "/extract/batch",
        }

        return JSONResponse(content=results)

    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract/specific")
async def extract_specific_info(
    file: UploadFile = File(...), field: str = Form("skills")
):
    """
    Extract specific field from resume

    Args:
        file: Uploaded resume file
        field: Specific field to extract (name, email, phone, skills, education, experience, certifications, languages)
    """
    try:
        # Validate field
        valid_fields = [
            "name",
            "email",
            "phone",
            "skills",
            "education",
            "experience",
            "certifications",
            "languages",
        ]
        if field not in valid_fields:
            raise HTTPException(
                status_code=400, detail=f"Invalid field. Must be one of: {valid_fields}"
            )

        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in [".pdf", ".txt", ".docx"]:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_extension} not supported. Allowed: .pdf, .txt, .docx",
            )

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=file_extension
        ) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name

        try:
            # Process resume using pipeline
            result = process_resume_pipeline(temp_file_path, False, None)

            if "error" in result:
                raise HTTPException(status_code=500, detail=result["error"])

            # Extract specific field
            extracted_value = result.get(field, "")

            return JSONResponse(
                content={
                    "filename": file.filename,
                    "field": field,
                    "extracted_value": extracted_value,
                    "metadata": result.get("metadata", {}),
                }
            )

        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    except Exception as e:
        logger.error(f"Error extracting specific field: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
async def get_status():
    """Get detailed system status"""
    try:
        # Test document processor
        processor_status = "healthy"
        try:
            _ = ResumeDocumentProcessor()
        except Exception as e:
            processor_status = f"unhealthy: {str(e)}"

        return {
            "status": "running",
            "version": "1.0.0",
            "components": {
                "document_processor": processor_status,
                "pipeline": "DocLing + LLM",
            },
            "timestamp": time.time(),
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "timestamp": time.time()}


if __name__ == "__main__":
    uvicorn.run("resume_extraction_server:app", host="0.0.0.0", port=8001, reload=False)
