#!/usr/bin/env python3
"""
Create Ground Truth Files from CSV
Loads resume data from CSV file and creates individual ground truth JSON files
"""

import json
import csv
import re
from pathlib import Path
from typing import Dict, Any


def parse_resume_text(resume_text: str) -> dict:
    """Parse resume text and extract structured information"""

    ground_truth = {
        "name": "",
        "email": "",
        "phone": "",
        "skills": [],
        "education": [],
        "experience": [],
        "certifications": [],
        "languages": [],
    }

    # Clean the text
    text = resume_text.strip()

    # Extract job title/name (appears at the beginning)
    lines = text.split("\n")
    for line in lines:
        line = line.strip()
        if (
            line
            and not line.startswith("Summary")
            and not line.startswith("Highlights")
        ):
            # Look for job titles in all caps
            if line.isupper() and len(line) > 5:
                ground_truth["name"] = line.strip()
                break

    # Extract skills from Highlights section
    skills = []
    highlights_match = re.search(
        r"Highlights\s+(.*?)(?=\s+Accomplishments|\s+Experience|$)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if highlights_match:
        highlights_text = highlights_match.group(1).strip()
        # Split by common delimiters and clean up
        skill_items = re.split(r"[,•\n]", highlights_text)
        for item in skill_items:
            item = item.strip()
            if item and len(item) > 2 and not item.startswith("Highlights"):
                skills.append(item)

    # Also look for specific skills mentioned
    skill_keywords = [
        "customer satisfaction",
        "team management",
        "marketing savvy",
        "conflict resolution",
        "training and development",
        "skilled multi-tasker",
        "client relations",
        "sales strategies",
        "inventory control",
        "loss prevention",
        "safety",
        "time management",
        "leadership",
    ]

    for keyword in skill_keywords:
        if keyword.lower() in text.lower():
            skills.append(keyword.title())

    # Remove duplicates and clean up
    skills = list(set([skill.strip() for skill in skills if skill.strip()]))
    ground_truth["skills"] = skills

    # Extract certifications from Accomplishments section
    certifications = []
    accomplishments_match = re.search(
        r"Accomplishments\s+(.*?)(?=\s+Experience|$)", text, re.DOTALL | re.IGNORECASE
    )
    if accomplishments_match:
        accomplishments_text = accomplishments_match.group(1).strip()
        # Look for certification patterns
        cert_patterns = [
            r"Training Certification",
            r"Certified by [^,]+",
            r"General Manager Training Certification",
            r"Accomplished Trainer",
        ]

        for pattern in cert_patterns:
            matches = re.findall(pattern, accomplishments_text, re.IGNORECASE)
            for match in matches:
                if match.strip():
                    certifications.append(match.strip())

    ground_truth["certifications"] = certifications

    # Extract experience from Experience section
    experience = []
    experience_match = re.search(
        r"Experience\s+(.*?)(?=\s+Summary|$)", text, re.DOTALL | re.IGNORECASE
    )
    if experience_match:
        experience_text = experience_match.group(1).strip()
        if experience_text:
            # Use the job title as the job title
            job_title = (
                ground_truth["name"]
                if ground_truth["name"]
                else "HR Administrator/Marketing Associate"
            )

            experience.append(
                {
                    "job_title": job_title,
                    "company": "Company Name",
                    "years_worked": "",
                    "description": experience_text,
                }
            )

    ground_truth["experience"] = experience

    return ground_truth


def create_ground_truth_from_csv(
    csv_file: str = "../../assets/resumes.csv",
) -> Dict[str, Any]:
    """Create ground truth files from CSV data"""
    print("🔧 Creating Ground Truth Files from CSV")
    print("=" * 60)

    # Create GT directory
    gt_dir = Path("../data/GT")
    gt_dir.mkdir(parents=True, exist_ok=True)

    # Create results directory
    results_dir = Path("../results")
    results_dir.mkdir(exist_ok=True)

    if not Path(csv_file).exists():
        print(f"❌ CSV file not found: {csv_file}")
        return {}

    print(f"📁 Reading CSV file: {csv_file}")
    print(f"📂 Ground truth directory: {gt_dir.absolute()}")

    created_files = 0
    all_ground_truth = {}

    try:
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                resume_id = row.get("ID", "").strip()
                resume_text = row.get("Resume_str", "").strip()

                if not resume_id or not resume_text:
                    print("⚠️ Skipping row with missing ID or text")
                    continue

                # Parse the resume text
                ground_truth = parse_resume_text(resume_text)

                # Save individual ground truth file
                gt_filename = f"{resume_id}_gt.json"
                gt_filepath = gt_dir / gt_filename

                with open(gt_filepath, "w", encoding="utf-8") as f:
                    json.dump(ground_truth, f, indent=2, ensure_ascii=False)

                created_files += 1
                print(f"✅ Created: {gt_filename}")

                # Store for main ground truth (use first one as template)
                if not all_ground_truth:
                    all_ground_truth = ground_truth.copy()

    except Exception as e:
        print(f"❌ Error reading CSV file: {e}")
        return {}

    # Save main ground truth file
    main_gt_file = results_dir / "ground_truth.json"
    with open(main_gt_file, "w", encoding="utf-8") as f:
        json.dump(all_ground_truth, f, indent=2, ensure_ascii=False)

    print("\n📊 Summary:")
    print(f"  📄 Total files processed: {created_files}")
    print(f"  ✅ Ground truth files created: {created_files}")
    print(f"  📂 Individual files location: {gt_dir.absolute()}")
    print(f"  📂 Main ground truth: {main_gt_file}")

    # Print sample extracted information
    if all_ground_truth:
        print("\n📋 Sample Extracted Information:")
        for field, value in all_ground_truth.items():
            if isinstance(value, list):
                print(f"  {field}: {len(value)} items")
            else:
                print(f"  {field}: {value}")

    return all_ground_truth


def create_sample_csv():
    """Create a sample CSV file for testing"""
    sample_data = [
        {
            "ID": "16852973",
            "Resume_str": "HR ADMINISTRATOR/MARKETING ASSOCIATE\n\nHR ADMINISTRATOR\n\nSummary\nDedicated Customer Service Manager with 15+ years of experience in Hospitality and Customer Service Management. Respected builder and leader of customer-focused teams; strives to instill a shared, enthusiastic commitment to customer service.\n\nHighlights\nFocused on customer satisfaction\nTeam management\nMarketing savvy\nConflict resolution techniques\nTraining and development\nSkilled multi-tasker\nClient relations specialist\n\nAccomplishments\nMissouri DOT Supervisor Training Certification\nCertified by IHG in Customer Loyalty and Marketing by Segment\nHilton Worldwide General Manager Training Certification\n\nExperience\nHR Administrator/Marketing Associate",
            "Resume_html": "<html>...</html>",
            "Category": "HR",
        }
    ]

    csv_file = Path("../assets/resumes.csv")
    csv_file.parent.mkdir(exist_ok=True)

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["ID", "Resume_str", "Resume_html", "Category"]
        )
        writer.writeheader()
        writer.writerows(sample_data)

    print(f"✅ Sample CSV file created: {csv_file}")
    return str(csv_file)


def main():
    """Main function"""
    print("🔧 Ground Truth Creation from CSV")
    print("=" * 60)

    # Check if CSV file exists, create sample if not
    csv_file = "../assets/resumes.csv"
    if not Path(csv_file).exists():
        print("📝 Creating sample CSV file...")
        csv_file = create_sample_csv()

    # Create ground truth files
    success = create_ground_truth_from_csv(csv_file)

    if success:
        print("\n🎉 Ground truth creation completed successfully!")
        return True
    else:
        print("\n❌ Ground truth creation failed!")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
