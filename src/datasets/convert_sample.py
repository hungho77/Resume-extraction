#!/usr/bin/env python3
"""
Sample PDF to Scanned PDF Converter
Converts a few PDFs to test the conversion process
"""

from pathlib import Path
from convert_to_scanned import PDFToScannedConverter


def main():
    """Convert a sample of PDFs"""
    print("🔄 Sample PDF to Scanned PDF Conversion")
    print("=" * 60)

    # Configuration
    input_directory = "../../data/INFORMATION-TECHNOLOGY"
    output_directory = "../../data/SCAN-INFORMATION-TECHNOLOGY"
    dpi = 150  # Lower resolution for smaller file sizes

    # Get first 5 PDF files
    input_path = Path(input_directory)
    pdf_files = list(input_path.glob("*.pdf"))[:5]  # Only first 5 files

    print(f"📁 Converting {len(pdf_files)} sample PDFs...")

    # Initialize converter
    converter = PDFToScannedConverter(output_directory)

    # Convert each PDF
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\n📄 Converting {i}/{len(pdf_files)}: {pdf_file.name}")

        result = converter.convert_pdf_to_scanned(str(pdf_file), dpi)

        if result.get("conversion_successful", False):
            print("  ✅ Successfully converted")
            print(f"  📊 Input size: {result.get('input_size_mb', 0)} MB")
            print(f"  📊 Output size: {result.get('output_size_mb', 0)} MB")
            print(f"  📊 Pages: {result.get('pages_converted', 0)}")
        else:
            print(f"  ❌ Conversion failed: {result.get('error', 'Unknown error')}")

    print("\n✅ Sample conversion completed!")


if __name__ == "__main__":
    main()
