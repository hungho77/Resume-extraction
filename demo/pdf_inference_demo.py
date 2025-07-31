#!/usr/bin/env python3
"""
PDF Inference Demo Script
Refactored to use pipeline code with input PDF file path and output JSON file
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.document_processor import ResumeDocumentProcessor


def process_pdf_to_json(pdf_path: str, output_path: str = None) -> Dict[str, Any]:
    """
    Process PDF file and extract structured information to JSON

    Args:
        pdf_path: Path to input PDF file
        output_path: Path to output JSON file (optional)

    Returns:
        Dictionary containing extracted entities and metadata
    """
    print("🧪 PDF to JSON Processing Pipeline")
    print("=" * 50)

    # Check if PDF file exists
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    print(f"📄 Input PDF: {pdf_file}")
    print(f"📄 File size: {pdf_file.stat().st_size} bytes")

    try:
        # Initialize document processor
        processor = ResumeDocumentProcessor()
        print("✅ Document processor initialized")

        # Process the PDF
        print("\n📝 Processing PDF...")
        result = processor.extract_entities_from_resume(str(pdf_file))

        if "error" in result:
            print(f"❌ Processing failed: {result['error']}")
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
                "file_path": metadata.get("file_path", str(pdf_file)),
                "file_size": metadata.get("file_size", pdf_file.stat().st_size),
                "text_length": metadata.get("text_length", 0),
                "processing_method": metadata.get("processing_method", "unknown"),
                "is_markdown_format": metadata.get("is_markdown_format", False),
                "completeness_score": validation.get("completeness_score", 0.0),
                "is_valid": validation.get("is_valid", False),
                "errors": validation.get("errors", []),
                "warnings": validation.get("warnings", []),
            },
        }

        # Print results summary
        print("\n✅ Processing completed successfully!")
        print(f"📊 File processed: {metadata.get('file_path', 'unknown')}")
        print(f"📊 File size: {metadata.get('file_size', 0)} bytes")
        print(f"📊 Text length: {metadata.get('text_length', 0)} characters")
        print(f"📊 Processing method: {metadata.get('processing_method', 'unknown')}")
        print(f"📊 Is markdown format: {metadata.get('is_markdown_format', False)}")
        print(f"📊 Completeness score: {validation.get('completeness_score', 0):.2f}")
        print(f"📊 Is valid: {validation.get('is_valid', False)}")
        print(f"📊 Errors: {len(validation.get('errors', []))}")
        print(f"📊 Warnings: {len(validation.get('warnings', []))}")

        # Print extracted fields
        print("\n📋 Extracted Information:")
        print(f"  👤 Name: {structured_output['name']}")
        print(f"  📧 Email: {structured_output['email']}")
        print(f"  📞 Phone: {structured_output['phone']}")
        print(f"  💻 Skills: {len(structured_output['skills'])} items")
        print(f"  🎓 Education: {len(structured_output['education'])} entries")
        print(f"  💼 Experience: {len(structured_output['experience'])} entries")
        print(f"  🏆 Certifications: {len(structured_output['certifications'])} items")
        print(f"  🌍 Languages: {len(structured_output['languages'])} items")

        # Save to JSON file if output path provided
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(structured_output, f, indent=2, ensure_ascii=False)

            print(f"\n💾 Results saved to: {output_file}")
            print(f"📄 Output file size: {output_file.stat().st_size} bytes")

        return structured_output

    except Exception as e:
        error_result = {
            "error": str(e),
            "metadata": {
                "file_path": str(pdf_file),
                "file_size": pdf_file.stat().st_size if pdf_file.exists() else 0,
            },
        }
        print(f"❌ Processing failed: {e}")
        return error_result


def batch_process_pdfs_to_json(
    pdf_directory: str, output_directory: str = None
) -> Dict[str, Any]:
    """
    Process multiple PDF files and save results to JSON files

    Args:
        pdf_directory: Directory containing PDF files
        output_directory: Directory to save JSON files (optional)

    Returns:
        Dictionary containing batch processing results
    """
    print("🧪 Batch PDF to JSON Processing Pipeline")
    print("=" * 50)

    pdf_dir = Path(pdf_directory)
    if not pdf_dir.exists() or not pdf_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {pdf_directory}")

    # Find all PDF files
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ No PDF files found in {pdf_directory}")
        return {"error": "No PDF files found"}

    print(f"📁 Processing directory: {pdf_dir}")
    print(f"📄 Found {len(pdf_files)} PDF files")

    results = {
        "total_files": len(pdf_files),
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

    # Process each PDF file
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\n📄 Processing {i}/{len(pdf_files)}: {pdf_file.name}")

        # Determine output file path
        if output_directory:
            output_dir = Path(output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"{pdf_file.stem}.json"
        else:
            output_file = pdf_file.with_suffix(".json")

        # Process the PDF
        try:
            result = process_pdf_to_json(str(pdf_file), str(output_file))

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

                results["summary"]["total_errors"] += len(metadata.get("errors", []))
                results["summary"]["total_warnings"] += len(
                    metadata.get("warnings", [])
                )

            else:
                results["failed_files"] += 1

            results["results"].append(
                {
                    "file": str(pdf_file),
                    "output": str(output_file),
                    "success": "error" not in result,
                    "result": result,
                }
            )

        except Exception as e:
            results["failed_files"] += 1
            results["results"].append(
                {
                    "file": str(pdf_file),
                    "output": str(output_file),
                    "success": False,
                    "error": str(e),
                }
            )
            print(f"❌ Failed to process {pdf_file.name}: {e}")

    # Calculate final statistics
    if results["successful_files"] > 0:
        results["summary"]["average_completeness"] /= results["successful_files"]

    # Print batch processing summary
    print("\n📊 Batch Processing Summary:")
    print(f"  📄 Total files: {results['total_files']}")
    print(f"  ✅ Successful: {results['successful_files']}")
    print(f"  ❌ Failed: {results['failed_files']}")
    print(
        f"  📊 Success rate: {(results['successful_files'] / results['total_files'] * 100):.1f}%"
    )
    print(f"  📋 Total entities found: {results['summary']['total_entities_found']}")
    print(
        f"  📊 Average completeness: {results['summary']['average_completeness']:.2f}"
    )
    print(f"  ⚠️ Total errors: {results['summary']['total_errors']}")
    print(f"  ⚠️ Total warnings: {results['summary']['total_warnings']}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="PDF to JSON Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single PDF file
  python pdf_inference_demo.py docs/examples_resume.pdf -o output.json
  
  # Process single PDF file (auto-generate output filename)
  python pdf_inference_demo.py docs/examples_resume.pdf
  
  # Batch process all PDFs in directory
  python pdf_inference_demo.py docs/ -o output_directory/
  
  # Batch process with auto-generated output files
  python pdf_inference_demo.py docs/
        """,
    )

    parser.add_argument("input", help="PDF file or directory to process")
    parser.add_argument(
        "-o", "--output", help="Output file or directory for JSON results"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Force batch processing mode (even for single file)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"❌ Input not found: {args.input}")
        sys.exit(1)

    try:
        if input_path.is_file() and not args.batch:
            # Single file processing
            result = process_pdf_to_json(str(input_path), args.output)
            if "error" in result:
                sys.exit(1)
        else:
            # Batch processing
            if input_path.is_file():
                # Single file in batch mode
                result = batch_process_pdfs_to_json(str(input_path.parent), args.output)
            else:
                # Directory processing
                result = batch_process_pdfs_to_json(str(input_path), args.output)

            if "error" in result:
                sys.exit(1)

        print("\n🎉 Processing completed successfully!")

    except Exception as e:
        print(f"❌ Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
