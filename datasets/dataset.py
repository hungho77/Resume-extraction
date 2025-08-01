#!/usr/bin/env python3
"""
Dataset Analysis Script for Resume PDFs
Analyzes PDFs in the INFORMATION-TECHNOLOGY folder to determine if they are scanned or readable
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.document_processor import ResumeDocumentProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFAnalyzer:
    """Analyzes PDFs to determine if they are scanned or readable"""

    def __init__(self):
        self.processor = ResumeDocumentProcessor()

    def analyze_pdf_readability(self, pdf_path: str) -> Dict[str, Any]:
        """
        Analyze a PDF to determine if it's scanned or readable

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Dictionary containing analysis results
        """
        try:
            pdf_file = Path(pdf_path)
            if not pdf_file.exists():
                return {"error": f"File not found: {pdf_path}"}

            # Get file info
            file_size = pdf_file.stat().st_size

            # Extract text using the processor
            text, used_smoldocling = self.processor.extract_text(str(pdf_file))

            # Analyze the extracted text
            analysis = {
                "file_path": str(pdf_file),
                "file_name": pdf_file.name,
                "file_size_bytes": file_size,
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "text_extracted": bool(text.strip()),
                "text_length": len(text.strip()) if text else 0,
                "used_smoldocling": used_smoldocling,
                "is_markdown_format": self.processor._is_markdown_format(text)
                if text
                else False,
                "readability_score": 0.0,
                "pdf_type": "unknown",
                "confidence": "low",
            }

            # Determine PDF type based on analysis
            if not text.strip():
                analysis["pdf_type"] = "scanned"
                analysis["confidence"] = "high"
                analysis["readability_score"] = 0.0
            elif used_smoldocling:
                analysis["pdf_type"] = "scanned"
                analysis["confidence"] = "high"
                analysis["readability_score"] = 0.3
            else:
                # Check if text seems readable
                readable_indicators = self._check_readability_indicators(text)
                analysis.update(readable_indicators)

                if readable_indicators["readability_score"] > 0.7:
                    analysis["pdf_type"] = "readable"
                    analysis["confidence"] = "high"
                elif readable_indicators["readability_score"] > 0.3:
                    analysis["pdf_type"] = "mixed"
                    analysis["confidence"] = "medium"
                else:
                    analysis["pdf_type"] = "scanned"
                    analysis["confidence"] = "medium"

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing {pdf_path}: {e}")
            return {"file_path": str(pdf_path), "error": str(e), "pdf_type": "error"}

    def _check_readability_indicators(self, text: str) -> Dict[str, Any]:
        """
        Check various indicators of text readability

        Args:
            text: Extracted text from PDF

        Returns:
            Dictionary with readability indicators
        """
        if not text:
            return {
                "readability_score": 0.0,
                "has_structured_content": False,
                "has_contact_info": False,
                "has_skills": False,
                "has_education": False,
                "has_experience": False,
                "avg_word_length": 0.0,
                "text_quality": "poor",
            }

        # Check for structured content (resume-like)
        has_structured_content = any(
            [
                "experience" in text.lower(),
                "education" in text.lower(),
                "skills" in text.lower(),
                "work" in text.lower(),
                "job" in text.lower(),
                "position" in text.lower(),
            ]
        )

        # Check for contact information
        has_contact_info = any(
            [
                "@" in text,  # Email
                any(char.isdigit() for char in text),  # Phone numbers
                "phone" in text.lower(),
                "email" in text.lower(),
            ]
        )

        # Check for skills section
        has_skills = any(
            [
                "skills" in text.lower(),
                "technologies" in text.lower(),
                "programming" in text.lower(),
                "languages" in text.lower(),
            ]
        )

        # Check for education section
        has_education = any(
            [
                "education" in text.lower(),
                "degree" in text.lower(),
                "university" in text.lower(),
                "college" in text.lower(),
                "bachelor" in text.lower(),
                "master" in text.lower(),
            ]
        )

        # Check for experience section
        has_experience = any(
            [
                "experience" in text.lower(),
                "work history" in text.lower(),
                "employment" in text.lower(),
                "career" in text.lower(),
            ]
        )

        # Calculate average word length
        words = text.split()
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0

        # Determine text quality
        if avg_word_length > 3 and has_structured_content:
            text_quality = "excellent"
        elif avg_word_length > 2.5 and (has_contact_info or has_skills):
            text_quality = "good"
        elif avg_word_length > 2:
            text_quality = "fair"
        else:
            text_quality = "poor"

        # Calculate readability score
        score = 0.0
        if has_structured_content:
            score += 0.3
        if has_contact_info:
            score += 0.2
        if has_skills:
            score += 0.15
        if has_education:
            score += 0.15
        if has_experience:
            score += 0.2
        if text_quality in ["excellent", "good"]:
            score += 0.1

        return {
            "readability_score": min(score, 1.0),
            "has_structured_content": has_structured_content,
            "has_contact_info": has_contact_info,
            "has_skills": has_skills,
            "has_education": has_education,
            "has_experience": has_experience,
            "avg_word_length": round(avg_word_length, 2),
            "text_quality": text_quality,
        }


def analyze_information_technology_dataset() -> Dict[str, Any]:
    """
    Analyze all PDFs in the INFORMATION-TECHNOLOGY folder

    Returns:
        Dictionary containing analysis results for all PDFs
    """
    print("🔍 Analyzing INFORMATION-TECHNOLOGY Dataset")
    print("=" * 60)

    # Define the dataset path
    dataset_path = Path("../data/INFORMATION-TECHNOLOGY")

    if not dataset_path.exists():
        print(f"❌ Dataset path not found: {dataset_path}")
        return {"error": f"Dataset path not found: {dataset_path}"}

    # Find all PDF files
    pdf_files = list(dataset_path.glob("*.pdf"))

    if not pdf_files:
        print(f"❌ No PDF files found in {dataset_path}")
        return {"error": "No PDF files found"}

    print(f"📁 Dataset path: {dataset_path}")
    print(f"📄 Found {len(pdf_files)} PDF files")

    # Initialize analyzer
    analyzer = PDFAnalyzer()

    # Analyze each PDF
    results = {
        "dataset_info": {
            "path": str(dataset_path),
            "total_files": len(pdf_files),
            "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "summary": {
            "scanned_pdfs": 0,
            "readable_pdfs": 0,
            "mixed_pdfs": 0,
            "error_pdfs": 0,
            "total_size_mb": 0.0,
            "avg_file_size_mb": 0.0,
        },
        "detailed_results": [],
    }

    total_size = 0.0

    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\n📄 Analyzing {i}/{len(pdf_files)}: {pdf_file.name}")

        # Analyze the PDF
        analysis = analyzer.analyze_pdf_readability(str(pdf_file))

        # Update summary
        if "error" in analysis:
            results["summary"]["error_pdfs"] += 1
        else:
            pdf_type = analysis.get("pdf_type", "unknown")
            if pdf_type == "scanned":
                results["summary"]["scanned_pdfs"] += 1
            elif pdf_type == "readable":
                results["summary"]["readable_pdfs"] += 1
            elif pdf_type == "mixed":
                results["summary"]["mixed_pdfs"] += 1

            total_size += analysis.get("file_size_mb", 0)

        results["detailed_results"].append(analysis)

        # Print progress
        if "error" in analysis:
            print(f"  ❌ Error: {analysis['error']}")
        else:
            print(
                f"  📊 Type: {analysis['pdf_type']} (confidence: {analysis['confidence']})"
            )
            print(f"  📊 Readability: {analysis['readability_score']:.2f}")
            print(f"  📊 Size: {analysis['file_size_mb']} MB")

    # Calculate summary statistics
    results["summary"]["total_size_mb"] = round(total_size, 2)
    results["summary"]["avg_file_size_mb"] = round(total_size / len(pdf_files), 2)

    # Print summary
    print("\n" + "=" * 60)
    print("📊 Dataset Analysis Summary")
    print("=" * 60)
    print(f"📄 Total PDFs: {len(pdf_files)}")
    print(f"📊 Scanned PDFs: {results['summary']['scanned_pdfs']}")
    print(f"📊 Readable PDFs: {results['summary']['readable_pdfs']}")
    print(f"📊 Mixed PDFs: {results['summary']['mixed_pdfs']}")
    print(f"❌ Error PDFs: {results['summary']['error_pdfs']}")
    print(f"💾 Total size: {results['summary']['total_size_mb']} MB")
    print(f"📊 Average file size: {results['summary']['avg_file_size_mb']} MB")

    # Calculate percentages
    total_valid = (
        results["summary"]["scanned_pdfs"]
        + results["summary"]["readable_pdfs"]
        + results["summary"]["mixed_pdfs"]
    )
    if total_valid > 0:
        scanned_pct = (results["summary"]["scanned_pdfs"] / total_valid) * 100
        readable_pct = (results["summary"]["readable_pdfs"] / total_valid) * 100
        mixed_pct = (results["summary"]["mixed_pdfs"] / total_valid) * 100

        print("\n📈 Percentages (excluding errors):")
        print(f"  📄 Scanned: {scanned_pct:.1f}%")
        print(f"  📄 Readable: {readable_pct:.1f}%")
        print(f"  📄 Mixed: {mixed_pct:.1f}%")

    return results


def save_analysis_results(
    results: Dict[str, Any], output_file: str = "dataset_analysis.json"
):
    """
    Save analysis results to JSON file

    Args:
        results: Analysis results dictionary
        output_file: Output file path
    """
    try:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Results saved to: {output_path}")
        print(f"📄 File size: {output_path.stat().st_size} bytes")

    except Exception as e:
        print(f"❌ Error saving results: {e}")


def main():
    """Main function"""
    print("🧪 PDF Dataset Analysis Tool")
    print("=" * 60)

    try:
        # Analyze the dataset
        results = analyze_information_technology_dataset()

        if "error" in results:
            print(f"❌ Analysis failed: {results['error']}")
            return False

        # Save results
        save_analysis_results(results)

        print("\n✅ Analysis completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
