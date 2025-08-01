#!/usr/bin/env python3
"""
Convert PDFs to Scanned PDFs
Converts readable PDFs to scanned PDFs by converting to images and back to PDFs
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any
import logging
import fitz  # PyMuPDF
from PIL import Image
import io

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFToScannedConverter:
    """Converts readable PDFs to scanned PDFs"""

    def __init__(self, output_dir: str = "../data/SCAN-INFORMATION-TECHNOLOGY"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def convert_pdf_to_scanned(self, pdf_path: str, dpi: int = 300) -> Dict[str, Any]:
        """
        Convert a PDF to a scanned PDF by converting to images and back

        Args:
            pdf_path: Path to the input PDF file
            dpi: Resolution for image conversion (default: 300)

        Returns:
            Dictionary containing conversion results
        """
        doc = None
        scanned_doc = None

        try:
            pdf_file = Path(pdf_path)
            if not pdf_file.exists():
                return {"error": f"File not found: {pdf_path}"}

            # Get file info
            file_size = pdf_file.stat().st_size

            # Open PDF with PyMuPDF
            doc = fitz.open(str(pdf_file))

            # Create output filename
            output_filename = f"SCAN_{pdf_file.stem}.pdf"
            output_path = self.output_dir / output_filename

            # Convert each page to image and back to PDF
            scanned_doc = fitz.open()

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)

                # Convert page to image with specified DPI
                mat = fitz.Matrix(dpi / 72, dpi / 72)  # Scale factor for DPI
                pix = page.get_pixmap(matrix=mat)

                # Convert to PIL Image
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))

                # Convert back to PDF page
                img_bytes = io.BytesIO()
                img.save(img_bytes, format="PNG")
                img_bytes.seek(0)

                # Insert image as PDF page
                scanned_page = scanned_doc.new_page(width=img.width, height=img.height)
                scanned_page.insert_image(
                    scanned_page.rect, stream=img_bytes.getvalue()
                )

                # Clean up pixmap
                pix = None

            # Save scanned PDF
            scanned_doc.save(str(output_path))

            # Get output file info
            output_size = output_path.stat().st_size

            return {
                "input_file": str(pdf_file),
                "output_file": str(output_path),
                "input_size_bytes": file_size,
                "output_size_bytes": output_size,
                "input_size_mb": round(file_size / (1024 * 1024), 2),
                "output_size_mb": round(output_size / (1024 * 1024), 2),
                "pages_converted": len(doc),
                "dpi_used": dpi,
                "conversion_successful": True,
            }

        except Exception as e:
            logger.error(f"Error converting {pdf_path}: {e}")
            return {
                "input_file": str(pdf_path),
                "error": str(e),
                "conversion_successful": False,
            }
        finally:
            # Clean up documents
            if doc:
                doc.close()
            if scanned_doc:
                scanned_doc.close()

    def batch_convert_pdfs(self, input_dir: str, dpi: int = 300) -> Dict[str, Any]:
        """
        Convert all PDFs in a directory to scanned PDFs

        Args:
            input_dir: Directory containing PDF files
            dpi: Resolution for image conversion

        Returns:
            Dictionary containing batch conversion results
        """
        print("🔄 Converting PDFs to Scanned PDFs")
        print("=" * 60)

        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"❌ Input directory not found: {input_dir}")
            return {"error": f"Input directory not found: {input_dir}"}

        # Find all PDF files
        pdf_files = list(input_path.glob("*.pdf"))
        if not pdf_files:
            print(f"❌ No PDF files found in {input_dir}")
            return {"error": "No PDF files found"}

        print(f"📁 Input directory: {input_path}")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"📄 Found {len(pdf_files)} PDF files")
        print(f"🎯 DPI setting: {dpi}")

        # Convert each PDF
        results = {
            "input_directory": str(input_path),
            "output_directory": str(self.output_dir),
            "total_files": len(pdf_files),
            "successful_conversions": 0,
            "failed_conversions": 0,
            "total_input_size_mb": 0.0,
            "total_output_size_mb": 0.0,
            "conversion_details": [],
        }

        for i, pdf_file in enumerate(pdf_files, 1):
            print(f"\n📄 Converting {i}/{len(pdf_files)}: {pdf_file.name}")

            # Convert the PDF
            result = self.convert_pdf_to_scanned(str(pdf_file), dpi)

            if result.get("conversion_successful", False):
                results["successful_conversions"] += 1
                results["total_input_size_mb"] += result.get("input_size_mb", 0)
                results["total_output_size_mb"] += result.get("output_size_mb", 0)

                print("  ✅ Successfully converted")
                print(f"  📊 Input size: {result.get('input_size_mb', 0)} MB")
                print(f"  📊 Output size: {result.get('output_size_mb', 0)} MB")
                print(f"  📊 Pages: {result.get('pages_converted', 0)}")
                print(f"  📁 Output: {Path(result.get('output_file', '')).name}")
            else:
                results["failed_conversions"] += 1
                print(f"  ❌ Conversion failed: {result.get('error', 'Unknown error')}")

            results["conversion_details"].append(result)

        # Print summary
        print("\n" + "=" * 60)
        print("📊 Conversion Summary")
        print("=" * 60)
        print(f"📄 Total files: {results['total_files']}")
        print(f"✅ Successful conversions: {results['successful_conversions']}")
        print(f"❌ Failed conversions: {results['failed_conversions']}")
        print(
            f"📊 Success rate: {(results['successful_conversions'] / results['total_files'] * 100):.1f}%"
        )
        print(f"💾 Total input size: {results['total_input_size_mb']:.2f} MB")
        print(f"💾 Total output size: {results['total_output_size_mb']:.2f} MB")

        if results["successful_conversions"] > 0:
            size_ratio = (
                results["total_output_size_mb"] / results["total_input_size_mb"]
            )
            print(f"📊 Size ratio (output/input): {size_ratio:.2f}x")

        return results


def save_conversion_results(
    results: Dict[str, Any], output_file: str = "conversion_results.json"
):
    """
    Save conversion results to JSON file

    Args:
        results: Conversion results dictionary
        output_file: Output file path
    """
    try:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            import json

            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Results saved to: {output_path}")
        print(f"📄 File size: {output_path.stat().st_size} bytes")

    except Exception as e:
        print(f"❌ Error saving results: {e}")


def main():
    """Main function"""
    print("🔄 PDF to Scanned PDF Converter")
    print("=" * 60)

    # Configuration
    input_directory = "../data/INFORMATION-TECHNOLOGY"
    output_directory = "../data/SCAN-INFORMATION-TECHNOLOGY"
    dpi = 100  # Even lower resolution for smaller file sizes

    try:
        # Initialize converter
        converter = PDFToScannedConverter(output_directory)

        # Convert PDFs
        results = converter.batch_convert_pdfs(input_directory, dpi)

        if "error" in results:
            print(f"❌ Conversion failed: {results['error']}")
            return False

        # Save results
        save_conversion_results(results)

        print("\n✅ Conversion completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
