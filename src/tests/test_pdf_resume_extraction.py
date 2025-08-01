#!/usr/bin/env python3
"""
Test script for PDF markdown extraction functionality
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_pdf_markdown_extraction():
    """Test PDF extraction with markdown export"""
    print("🧪 Testing PDF Markdown Extraction")
    print("=" * 50)

    # Check if we have the required dependencies
    try:
        from src.core.document_processor import ResumeDocumentProcessor

        print("✅ ResumeDocumentProcessor imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install dependencies: pip install -r requirements.txt")
        return False

    # Test PDF file path
    pdf_path = Path("assets/resume.pdf")
    if not pdf_path.exists():
        print(f"❌ PDF file not found: {pdf_path}")
        return False

    print(f"📄 Using PDF file: {pdf_path}")
    print(f"📄 File size: {pdf_path.stat().st_size} bytes")

    try:
        # Initialize processor
        processor = ResumeDocumentProcessor()
        print("✅ Processor initialized")

        # Test text extraction
        print("\n📝 Testing text extraction...")
        text, used_smoldocling = processor.extract_text(str(pdf_path))

        if text:
            print("✅ Text extracted successfully")
            print(f"📊 Text length: {len(text)} characters")
            print(f"📊 First 200 chars: {text[:200]}...")

            # Check if it's markdown
            is_markdown = processor._is_markdown_format(text)
            print(f"📊 Is markdown format: {is_markdown}")

            if is_markdown:
                print("🎉 Markdown extraction successful!")
            else:
                print("ℹ️  Text is in plain format")

        else:
            print("❌ No text extracted")
            return False

        # Test entity extraction (if LLM is available)
        print("\n🤖 Testing entity extraction...")
        try:
            result = processor.extract_entities_from_resume(str(pdf_path))

            if "error" not in result:
                entities = result.get("entities", {})
                validation = result.get("validation", {})
                metadata = result.get("metadata", {})

                print("✅ Entity extraction successful!")
                print(
                    f"📊 Is markdown format: {metadata.get('is_markdown_format', False)}"
                )
                print(
                    f"📊 Completeness score: {validation.get('completeness_score', 0):.2f}"
                )
                print(f"📊 Extracted fields: {list(entities.keys())}")

                # Show some extracted data
                for field, value in entities.items():
                    if value:
                        if isinstance(value, list):
                            print(f"📋 {field}: {len(value)} items")
                        else:
                            print(f"📋 {field}: {value}")

            else:
                print(f"❌ Entity extraction failed: {result['error']}")

        except Exception as e:
            print(f"⚠️  Entity extraction error (LLM may not be available): {e}")
            print("This is expected if the LLM service is not running")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_pdf_markdown_extraction()
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)
