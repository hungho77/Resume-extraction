#!/usr/bin/env python3
"""
Gradio Demo for Resume Extraction
Self-contained application that processes resumes directly
"""

import gradio as gr
import json
import tempfile
from pathlib import Path
from typing import Dict, Any
import time
import os
import sys

# Add the src directory to the path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.core.document_processor import ResumeDocumentProcessor
from src.core.config import DocLingConfig

# Default configuration
DEFAULT_MODELS = [
    "Qwen/Qwen3-8B",
    "gpt-3.5-turbo",
    "gpt-4",
    "claude-3-sonnet",
    "claude-3-opus",
]


class ResumeExtractionDemo:
    """Self-contained Gradio demo for resume extraction"""

    def __init__(self):
        self.document_processor = None
        self.current_config = None

    def create_processor(
        self, model: str, use_ocr: bool, api_key: str = ""
    ) -> ResumeDocumentProcessor:
        """Create a document processor with custom settings"""
        config = DocLingConfig()

        # Update model if it's a cloud model
        if model in ["gpt-3.5-turbo", "gpt-4", "claude-3-sonnet", "claude-3-opus"]:
            config.llm_model = model
            if api_key:
                config.openai_api_key = api_key
        else:
            # Local model
            config.llm_model = model

        # Update OCR setting
        config.use_ocr = use_ocr

        # Create new processor with custom config
        processor = ResumeDocumentProcessor()
        processor.config = config

        return processor

    def test_connection(self, model: str, api_key: str) -> str:
        """Test if the selected model is available"""
        try:
            # Create a test processor
            _ = self.create_processor(model, True, api_key)

            # Test basic functionality
            if model in ["gpt-3.5-turbo", "gpt-4", "claude-3-sonnet", "claude-3-opus"]:
                if not api_key:
                    return f"❌ API Key Required\nFor {model}, you need to provide an API key"
                return f"✅ Model {model} configured\nAPI key provided"
            else:
                return f"✅ Local model {model} configured\nMake sure vLLM server is running"

        except Exception as e:
            return f"❌ Configuration Error: {str(e)}"

    def extract_resume(
        self,
        pdf_file,
        api_key: str = "",
        model: str = "Qwen/Qwen3-8B",
        use_ocr: bool = True,
    ) -> tuple[str, str, str]:
        """
        Extract resume information directly using the document processor

        Args:
            pdf_file: Uploaded PDF file
            api_key: API key for cloud services
            model: Model to use for extraction
            use_ocr: Whether to use OCR for scanned documents

        Returns:
            Tuple of (extracted_html, json_output, download_filename)
        """
        if not pdf_file:
            return "❌ Please upload a PDF file", "", ""

        try:
            # Get file information
            if hasattr(pdf_file, "read"):
                # Standard file object
                file_content = pdf_file.read()
                file_name = pdf_file.name
            elif hasattr(pdf_file, "name"):
                # Gradio file object - get the actual file path
                file_path = pdf_file.name
                file_name = Path(file_path).name
                # Read the file content
                with open(file_path, "rb") as f:
                    file_content = f.read()
            else:
                return "❌ Invalid file object", "", ""

            # Create temporary file for processing
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name

            try:
                # Create processor with custom settings
                processor = self.create_processor(model, use_ocr, api_key)

                print(f"🔗 Processing resume with model: {model}")
                print(f"📄 File: {file_name}")
                print(f"👁️ Use OCR: {use_ocr}")

                # Process the resume
                start_time = time.time()
                result = processor.extract_entities_from_resume(temp_file_path)
                end_time = time.time()

                if "error" in result:
                    error_msg = f"❌ Processing Error: {result['error']}"
                    print(error_msg)
                    return error_msg, "", ""

                # Format the extracted information for display as HTML
                extracted_html = self._format_extraction_result_html(result)

                # Create JSON output for download
                json_output = json.dumps(result, indent=2, ensure_ascii=False)

                # Create download filename using the original file name
                download_filename = f"{Path(file_name).stem}_extracted.json"

                processing_time = end_time - start_time
                print(f"✅ Extraction completed in {processing_time:.2f}s")
                print(f"📄 Original file: {file_name}")
                print(f"📄 Output file: {download_filename}")

                return extracted_html, json_output, download_filename

            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)

        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            print(error_msg)
            return error_msg, "", ""

    def _format_extraction_result_html(self, result: Dict[str, Any]) -> str:
        """Format extraction result as beautiful HTML"""
        if "error" in result:
            return f"""
            <div style="background: #f8d7da; border: 1px solid #f5c6cb; border-radius: 8px; padding: 15px; color: #721c24;">
                <h3 style="margin: 0 0 10px 0;">❌ Error</h3>
                <p style="margin: 0;">{result["error"]}</p>
            </div>
            """

        # Debug: Print the actual result structure
        print(f"🔍 Result structure: {list(result.keys())}")

        # Handle both flat structure (direct keys) and nested structure (entities object)
        if "entities" in result:
            # Nested structure
            entities = result.get("entities", {})
            metadata = result.get("metadata", {})
            validation = result.get("validation", {})
        else:
            # Flat structure - the result itself contains the entities
            entities = result
            metadata = result.get("metadata", {})
            validation = result.get("validation", {})

        print(f"🔍 Entities: {list(entities.keys()) if entities else 'None'}")
        print(f"🔍 Metadata: {list(metadata.keys()) if metadata else 'None'}")
        print(f"🔍 Validation: {list(validation.keys()) if validation else 'None'}")

        # Debug: Show actual values
        print(f"🔍 Name: '{entities.get('name', '')}'")
        print(f"🔍 Email: '{entities.get('email', '')}'")
        print(f"🔍 Phone: '{entities.get('phone', '')}'")
        print(f"🔍 Skills: {entities.get('skills', [])}")
        print(f"🔍 Education: {entities.get('education', [])}")
        print(f"🔍 Experience: {entities.get('experience', [])}")

        # Start building HTML
        html = """
        <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
        """

        # Personal Information Card
        name = entities.get("name", "")
        email = entities.get("email", "")
        phone = entities.get("phone", "")

        html += f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; padding: 20px; margin: 15px 0; color: white; box-shadow: 0 8px 32px rgba(0,0,0,0.1);">
            <h2 style="margin: 0 0 15px 0; font-size: 24px;">👤 Personal Information</h2>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                <div>
                    <strong>Name:</strong><br>
                    <span style="font-size: 18px;">{name if name else "Not found"}</span>
                </div>
                <div>
                    <strong>Email:</strong><br>
                    <span style="font-size: 16px;">{email if email else "Not found"}</span>
                </div>
                <div>
                    <strong>Phone:</strong><br>
                    <span style="font-size: 16px;">{phone if phone else "Not found"}</span>
                </div>
            </div>
        </div>
        """

        # Skills Section
        skills = entities.get("skills", [])
        if skills:
            skills_html = ""
            for skill in skills:  # Show all skills
                skills_html += f'<span class="skill-tag">{skill}</span>'

            html += f"""
            <div style="background: white; border-radius: 10px; padding: 20px; margin: 15px 0; box-shadow: 0 4px 16px rgba(0,0,0,0.1);">
                <h3 style="margin: 0 0 15px 0; color: #333;">💻 Skills ({len(skills)} items)</h3>
                <div style="line-height: 1.6;">
                    {skills_html}
                </div>
            </div>
            """

        # Education Section
        education = entities.get("education", [])
        if education:
            edu_html = ""
            for edu in education:
                degree = edu.get("degree", "Unknown")
                institution = edu.get("institution", "Unknown")
                year = edu.get("graduation_year", "Unknown")
                edu_html += f"""
                <div style="border-left: 3px solid #667eea; padding-left: 15px; margin: 10px 0;">
                    <strong>{degree}</strong><br>
                    <span style="color: #666;">{institution} ({year})</span>
                </div>
                """

            html += f"""
            <div style="background: white; border-radius: 10px; padding: 20px; margin: 15px 0; box-shadow: 0 4px 16px rgba(0,0,0,0.1);">
                <h3 style="margin: 0 0 15px 0; color: #333;">🎓 Education ({len(education)} entries)</h3>
                {edu_html}
            </div>
            """

        # Experience Section
        experience = entities.get("experience", [])
        if experience:
            exp_html = ""
            for exp in experience:
                title = exp.get("job_title", "Unknown")
                company = exp.get("company", "Unknown")
                years = exp.get("years_worked", "Unknown")
                description = exp.get("description", "")
                exp_html += f"""
                <div style="border-left: 3px solid #667eea; padding-left: 15px; margin: 15px 0;">
                    <strong style="font-size: 16px;">{title}</strong><br>
                    <span style="color: #667eea; font-weight: 500;">{company}</span><br>
                    <span style="color: #666; font-size: 14px;">{years}</span>
                    {f'<p style="margin: 10px 0 0 0; color: #555; font-size: 14px; line-height: 1.5;">{description}</p>' if description else ""}
                </div>
                """

            html += f"""
            <div style="background: white; border-radius: 10px; padding: 20px; margin: 15px 0; box-shadow: 0 4px 16px rgba(0,0,0,0.1);">
                <h3 style="margin: 0 0 15px 0; color: #333;">💼 Experience ({len(experience)} entries)</h3>
                {exp_html}
            </div>
            """

        # Certifications Section
        certifications = entities.get("certifications", [])
        if certifications:
            cert_html = ""
            for cert in certifications:
                cert_html += f'<div style="padding: 5px 0;">• {cert}</div>'

            html += f"""
            <div style="background: white; border-radius: 10px; padding: 20px; margin: 15px 0; box-shadow: 0 4px 16px rgba(0,0,0,0.1);">
                <h3 style="margin: 0 0 15px 0; color: #333;">🏆 Certifications ({len(certifications)} items)</h3>
                {cert_html}
            </div>
            """

        # Languages Section
        languages = entities.get("languages", [])
        if languages:
            lang_html = ""
            for lang in languages:
                lang_html += f'<span class="skill-tag">{lang}</span>'

            html += f"""
            <div style="background: white; border-radius: 10px; padding: 20px; margin: 15px 0; box-shadow: 0 4px 16px rgba(0,0,0,0.1);">
                <h3 style="margin: 0 0 15px 0; color: #333;">🌍 Languages ({len(languages)} items)</h3>
                <div style="line-height: 1.6;">
                    {lang_html}
                </div>
            </div>
            """

        # Warning if LLM is not available (only show this critical error)
        if not metadata.get("llm_available", True):
            html += """
            <div style="background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px; padding: 15px; margin: 15px 0; color: #856404;">
                <h4 style="margin: 0 0 10px 0;">⚠️ WARNING: LLM service is not available!</h4>
                <p style="margin: 0 0 10px 0;">This means the API could not connect to the language model.</p>
                <ul style="margin: 0; padding-left: 20px;">
                    <li>Is the vLLM server running?</li>
                    <li>Is the API key correct (for cloud services)?</li>
                    <li>Is the model name correct?</li>
                </ul>
            </div>
            """

        html += "</div>"
        return html

    def _format_extraction_result(self, result: Dict[str, Any]) -> str:
        """Format extraction result for display"""
        if "error" in result:
            return f"❌ Error: {result['error']}"

        # Debug: Print the actual result structure
        print(f"🔍 Result structure: {list(result.keys())}")

        # Handle both flat structure (direct keys) and nested structure (entities object)
        if "entities" in result:
            # Nested structure
            entities = result.get("entities", {})
            metadata = result.get("metadata", {})
            validation = result.get("validation", {})
        else:
            # Flat structure - the result itself contains the entities
            entities = result
            metadata = result.get("metadata", {})
            validation = result.get("validation", {})

        print(f"🔍 Entities: {list(entities.keys()) if entities else 'None'}")
        print(f"🔍 Metadata: {list(metadata.keys()) if metadata else 'None'}")
        print(f"🔍 Validation: {list(validation.keys()) if validation else 'None'}")

        # Debug: Show actual values
        print(f"🔍 Name: '{entities.get('name', '')}'")
        print(f"🔍 Email: '{entities.get('email', '')}'")
        print(f"🔍 Phone: '{entities.get('phone', '')}'")
        print(f"🔍 Skills: {entities.get('skills', [])}")
        print(f"🔍 Education: {entities.get('education', [])}")
        print(f"🔍 Experience: {entities.get('experience', [])}")

        # Format basic information
        output = "📋 Resume Extraction Results\n"
        output += "=" * 50 + "\n\n"

        # Show the actual JSON structure for debugging
        output += "🔍 Raw JSON Structure:\n"
        output += "-" * 25 + "\n"
        output += f"Keys in result: {list(result.keys())}\n"
        if "entities" in result:
            output += f"Entity keys: {list(result['entities'].keys())}\n"
        else:
            output += f"Direct entity keys: {[k for k in result.keys() if k not in ['metadata', 'api_metadata', 'validation']]}\n"
        output += "\n"

        # Personal Information
        output += "👤 Personal Information\n"
        output += "-" * 25 + "\n"
        name = entities.get("name", "")
        email = entities.get("email", "")
        phone = entities.get("phone", "")
        output += f"Name: {name if name else 'Not found'}\n"
        output += f"Email: {email if email else 'Not found'}\n"
        output += f"Phone: {phone if phone else 'Not found'}\n\n"

        # Skills
        skills = entities.get("skills", [])
        output += f"💻 Skills ({len(skills)} items)\n"
        output += "-" * 15 + "\n"
        if skills:
            for skill in skills[:10]:  # Show first 10 skills
                output += f"• {skill}\n"
            if len(skills) > 10:
                output += f"... and {len(skills) - 10} more\n"
        else:
            output += "No skills found\n"
        output += "\n"

        # Education
        education = entities.get("education", [])
        output += f"🎓 Education ({len(education)} entries)\n"
        output += "-" * 20 + "\n"
        for edu in education:
            degree = edu.get("degree", "Unknown")
            institution = edu.get("institution", "Unknown")
            year = edu.get("graduation_year", "Unknown")
            output += f"• {degree} from {institution} ({year})\n"
        if not education:
            output += "No education found\n"
        output += "\n"

        # Experience
        experience = entities.get("experience", [])
        output += f"💼 Experience ({len(experience)} entries)\n"
        output += "-" * 20 + "\n"
        for exp in experience:
            title = exp.get("job_title", "Unknown")
            company = exp.get("company", "Unknown")
            years = exp.get("years_worked", "Unknown")
            description = exp.get("description", "")
            output += f"• {title} at {company} ({years})\n"
            if description:
                output += (
                    f"  {description[:100]}{'...' if len(description) > 100 else ''}\n"
                )
        if not experience:
            output += "No experience found\n"
        output += "\n"

        # Certifications
        certifications = entities.get("certifications", [])
        output += f"🏆 Certifications ({len(certifications)} items)\n"
        output += "-" * 25 + "\n"
        for cert in certifications:
            output += f"• {cert}\n"
        if not certifications:
            output += "No certifications found\n"
        output += "\n"

        # Languages
        languages = entities.get("languages", [])
        output += f"🌍 Languages ({len(languages)} items)\n"
        output += "-" * 20 + "\n"
        for lang in languages:
            output += f"• {lang}\n"
        if not languages:
            output += "No languages found\n"
        output += "\n"

        # Processing Information
        output += "🔧 Processing Information\n"
        output += "-" * 25 + "\n"
        output += f"File: {metadata.get('file_path', 'Unknown')}\n"
        output += f"File Size: {metadata.get('file_size', 0)} bytes\n"
        output += f"Text Length: {metadata.get('text_length', 0)} characters\n"
        output += f"Processing Method: {metadata.get('processing_method', 'Unknown')}\n"
        output += f"Markdown Format: {metadata.get('is_markdown_format', False)}\n"
        output += f"OCR Enabled: {metadata.get('ocr_enabled', False)}\n"
        output += (
            f"SmolDocLing Available: {metadata.get('smoldocling_available', False)}\n"
        )
        output += f"LLM Available: {metadata.get('llm_available', False)}\n"

        # Validation Results
        completeness = validation.get("completeness_score", 0.0)
        is_valid = validation.get("is_valid", False)
        errors = validation.get("errors", [])
        warnings = validation.get("warnings", [])

        output += f"Completeness Score: {completeness:.2f}\n"
        output += f"Valid: {is_valid}\n"
        output += f"Errors: {len(errors)}\n"
        output += f"Warnings: {len(warnings)}\n"

        # Add warning if LLM is not available
        if not metadata.get("llm_available", True):
            output += "\n⚠️ WARNING: LLM service is not available!\n"
            output += "This means the API could not connect to the language model.\n"
            output += "Please check:\n"
            output += "• Is the vLLM server running?\n"
            output += "• Is the API key correct (for cloud services)?\n"
            output += "• Is the model name correct?\n"

        if errors:
            output += "\n❌ Errors:\n"
            for error in errors:
                output += f"• {error}\n"

        if warnings:
            output += "\n⚠️ Warnings:\n"
            for warning in warnings:
                output += f"• {warning}\n"

        return output


def create_demo() -> gr.Blocks:
    """Create the Gradio demo interface"""

    demo_instance = ResumeExtractionDemo()

    with gr.Blocks(
        title="Resume Extraction Demo",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1400px !important;
        }
        .resume-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            padding: 20px;
            margin: 10px 0;
            color: white;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        .info-section {
            background: rgba(255,255,255,0.95);
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #667eea;
        }
        .skill-tag {
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 4px 8px;
            border-radius: 12px;
            margin: 2px;
            font-size: 12px;
        }
        .status-success {
            color: #28a745;
            font-weight: bold;
        }
        .status-error {
            color: #dc3545;
            font-weight: bold;
        }
        .status-warning {
            color: #ffc107;
            font-weight: bold;
        }
        .results-container {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
        }
        .accordion-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 8px;
        }
        .extract-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            color: white;
            font-weight: bold;
        }
        .extract-btn:hover {
            background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
        }
        """,
    ) as demo:
        gr.Markdown("""
        # 📄 Resume Extraction Demo
        
        <div style="text-align: center; margin: 20px 0;">
            <p style="font-size: 18px; color: #666;">
                Extract structured information from PDF resumes using advanced AI models
            </p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Accordion("⚙️ Model Configuration", open=False):
                    api_key_input = gr.Textbox(
                        label="API Key (Optional)",
                        placeholder="Enter your API key for cloud services",
                        type="password",
                        info="API key for OpenAI, Anthropic, or other cloud services",
                    )

                    model_dropdown = gr.Dropdown(
                        label="Model",
                        choices=DEFAULT_MODELS,
                        value="Qwen/Qwen3-8B",
                        info="Choose the AI model for extraction",
                    )

                    use_ocr_checkbox = gr.Checkbox(
                        label="Use OCR for Scanned Documents",
                        value=True,
                        info="Enable SmolDocLing OCR for scanned PDFs",
                    )

                    test_connection_btn = gr.Button(
                        "🔗 Test Configuration", variant="secondary", size="sm"
                    )
                    connection_status = gr.Textbox(
                        label="Configuration Status",
                        interactive=False,
                        lines=2,
                        elem_classes=["status-success"],
                    )

            with gr.Column(scale=1):
                with gr.Group(elem_classes=["info-section"]):
                    gr.Markdown("### 📄 Resume Upload")

                    pdf_upload = gr.File(
                        label="Upload PDF Resume",
                        file_types=[".pdf"],
                        file_count="single",
                        height=120,
                    )

                    extract_btn = gr.Button(
                        "🚀 Extract Resume",
                        variant="primary",
                        size="lg",
                        elem_classes=["extract-btn"],
                    )

                    download_btn = gr.Button(
                        "📥 Download JSON", variant="secondary", size="sm"
                    )

        with gr.Row():
            gr.Markdown("### 📋 Extraction Results")

        with gr.Row():
            results_output = gr.HTML(
                label="Resume Information", elem_classes=["results-container"]
            )

        # Hidden components for JSON data
        json_output = gr.Textbox(visible=False)
        download_filename = gr.Textbox(visible=False)

        # Event handlers
        def test_connection(model, api_key):
            return demo_instance.test_connection(model, api_key)

        def extract_resume(pdf_file, api_key, model, use_ocr):
            if not pdf_file:
                return (
                    """
                <div style="background: #f8d7da; border: 1px solid #f5c6cb; border-radius: 8px; padding: 15px; color: #721c24;">
                    <h3 style="margin: 0;">❌ Error</h3>
                    <p style="margin: 10px 0 0 0;">Please upload a PDF file</p>
                </div>
                """,
                    "",
                    "",
                )

            extracted_html, json_data, download_filename = demo_instance.extract_resume(
                pdf_file, api_key, model, use_ocr
            )

            return extracted_html, json_data, download_filename

        def download_json(json_data, filename):
            if not json_data:
                return None

            # Create temporary file for download
            temp_file = tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, encoding="utf-8"
            )
            temp_file.write(json_data)
            temp_file.close()

            return temp_file.name

        # Connect events
        test_connection_btn.click(
            test_connection,
            inputs=[model_dropdown, api_key_input],
            outputs=[connection_status],
        )

        extract_btn.click(
            extract_resume,
            inputs=[
                pdf_upload,
                api_key_input,
                model_dropdown,
                use_ocr_checkbox,
            ],
            outputs=[results_output, json_output, download_filename],
        )

        download_btn.click(
            download_json,
            inputs=[json_output, download_filename],
            outputs=[gr.File(label="Download JSON")],
        )

        # Auto-update connection status when model changes
        model_dropdown.change(
            test_connection,
            inputs=[model_dropdown, api_key_input],
            outputs=[connection_status],
        )

        gr.Markdown("""
        ## 📊 Supported Output Fields
        - **Personal Info**: Name, Email, Phone
        - **Skills**: Technical and professional skills
        - **Education**: Degree, institution, graduation year
        - **Experience**: Job title, company, years worked, description
        - **Certifications**: Professional certifications
        - **Languages**: Spoken/written languages
        """)

    return demo


def main():
    """Main function to launch the demo"""
    print("🚀 Starting Resume Extraction Gradio Demo")
    print("=" * 50)

    # Create and launch the demo
    demo = create_demo()

    # Launch with configuration
    demo.launch(
        server_name="0.0.0.0",
        server_port=8080,
        share=True,
        show_error=True,
        quiet=False,
    )


if __name__ == "__main__":
    main()
