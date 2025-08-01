#!/usr/bin/env python3
"""
Resume Parser Evaluation Script
Evaluates the accuracy of resume parsing by comparing extracted entities with ground truth
"""

import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Any
import logging
from fuzzywuzzy import fuzz
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResumeEvaluator:
    """Evaluates resume parsing accuracy against ground truth"""

    def __init__(self, ground_truth_file: str = "../results/ground_truth.json"):
        self.ground_truth_file = Path(ground_truth_file)
        self.ground_truth = self._parse_ground_truth()

    def _parse_ground_truth(self) -> Dict[str, Any]:
        """Parse ground truth from JSON file"""
        if not self.ground_truth_file.exists():
            logger.error(f"Ground truth file not found: {self.ground_truth_file}")
            return {}

        try:
            with open(self.ground_truth_file, "r", encoding="utf-8") as f:
                ground_truth = json.load(f)
            logger.info(f"Loaded ground truth with {len(ground_truth)} fields")
            return ground_truth

        except Exception as e:
            logger.error(f"Error parsing ground truth: {e}")
            return {}

    def _extract_entities_from_text(self, text: str) -> Dict[str, Any]:
        """Extract entities from the ground truth text"""
        entities = {
            "name": "",
            "email": "",
            "phone": "",
            "skills": [],
            "education": [],
            "experience": [],
            "certifications": [],
            "languages": [],
        }

        # Extract job title (appears at the beginning)
        job_title_match = re.match(r"^([A-Z\s]+)", text)
        if job_title_match:
            entities["name"] = job_title_match.group(1).strip()

        # Extract skills (appears at the end after "Skills")
        skills_section = re.search(r"Skills\s+(.+)$", text, re.DOTALL)
        if skills_section:
            skills_text = skills_section.group(1).strip()
            # Split by commas and clean up
            skills = [
                skill.strip() for skill in skills_text.split(",") if skill.strip()
            ]
            entities["skills"] = skills

        # Extract education - look for "Education" section
        education_section = re.search(
            r"Education\s+(.+?)(?=\s+Certifications|\s+Skills|$)", text, re.DOTALL
        )
        if education_section:
            education_text = education_section.group(1).strip()
            # Parse education details
            education = self._parse_education(education_text)
            entities["education"] = education

        # Extract experience - look for "Experience" section
        experience_section = re.search(
            r"Experience\s+(.+?)(?=\s+Education|\s+Certifications|\s+Skills|$)",
            text,
            re.DOTALL,
        )
        if experience_section:
            experience_text = experience_section.group(1).strip()
            # Parse experience details
            experience = self._parse_experience(experience_text)
            entities["experience"] = experience

        # Extract certifications - look for "Certifications" section
        certifications_section = re.search(
            r"Certifications\s+(.+?)(?=\s+Skills|$)", text, re.DOTALL
        )
        if certifications_section:
            cert_text = certifications_section.group(1).strip()
            # Split by commas and clean up
            certifications = [
                cert.strip() for cert in cert_text.split(",") if cert.strip()
            ]
            entities["certifications"] = certifications

        # For the specific results.txt file, manually extract based on known structure
        # The text has a specific format, so let's parse it more carefully

        # Extract skills from the end of the text
        if "Skills" in text:
            skills_start = text.rfind("Skills")
            skills_text = text[skills_start:].replace("Skills", "").strip()
            skills = [
                skill.strip() for skill in skills_text.split(",") if skill.strip()
            ]
            entities["skills"] = skills

        # Extract education information
        if "Bachelor of Science" in text and "Florida International" in text:
            entities["education"] = [
                {
                    "degree": "Bachelor of Science",
                    "institution": "Florida International University",
                    "graduation_year": "2005",
                }
            ]

        # Extract certifications
        if "CompTIA Network+" in text:
            entities["certifications"] = ["CompTIA Network+ - 2014"]

        # Extract experience - look for job titles in all caps
        experience_entries = []

        # Information Technology Technician I
        if "Information Technology Technician I" in text:
            experience_entries.append(
                {
                    "job_title": "Information Technology Technician I",
                    "company": "Company Name",
                    "years_worked": "Aug 2007 to Current",
                    "description": "Migrating and managing user accounts in Microsoft Office 365 and Exchange Online...",
                }
            )

        # Information Services Liaison
        if "Information Services Liaison" in text:
            experience_entries.append(
                {
                    "job_title": "Information Services Liaison",
                    "company": "Company Name",
                    "years_worked": "Aug 2005 to Aug 2007",
                    "description": "Troubleshooting hardware and software problems over the telephone...",
                }
            )

        entities["experience"] = experience_entries

        return entities

    def _parse_education(self, education_text: str) -> List[Dict[str, str]]:
        """Parse education section"""
        education = []

        # Look for degree, institution, and year patterns
        degree_match = re.search(
            r"(Bachelor|Master|PhD|B\.S\.|M\.S\.|Ph\.D\.)", education_text
        )
        institution_match = re.search(r"(University|College|Institute)", education_text)
        year_match = re.search(r"(\d{4})", education_text)

        if degree_match or institution_match:
            education.append(
                {
                    "degree": degree_match.group(1) if degree_match else "",
                    "institution": institution_match.group(1)
                    if institution_match
                    else "",
                    "graduation_year": year_match.group(1) if year_match else "",
                }
            )

        return education

    def _parse_experience(self, experience_text: str) -> List[Dict[str, str]]:
        """Parse experience section"""
        experience = []

        # Split by job titles (all caps followed by text)
        job_sections = re.split(r"([A-Z\s]+)\s+", experience_text)

        for i in range(1, len(job_sections), 2):
            if i + 1 < len(job_sections):
                job_title = job_sections[i].strip()
                job_description = job_sections[i + 1].strip()

                # Extract years worked
                years_match = re.search(r"(\d{4}\s+to\s+\w+)", job_description)
                years_worked = years_match.group(1) if years_match else ""

                experience.append(
                    {
                        "job_title": job_title,
                        "company": "Company Name",  # Default as not specified in text
                        "years_worked": years_worked,
                        "description": job_description,
                    }
                )

        return experience

    def evaluate_entity_accuracy(self, predicted: str, ground_truth: str) -> float:
        """Evaluate accuracy of string entities using fuzzy matching"""
        if not ground_truth and not predicted:
            return 1.0  # Both empty is perfect match
        if not ground_truth or not predicted:
            return 0.0  # One empty, one not is no match

        # Use fuzzy string matching
        ratio = fuzz.ratio(predicted.lower(), ground_truth.lower()) / 100.0
        return ratio

    def evaluate_list_accuracy(
        self, predicted: List[str], ground_truth: List[str]
    ) -> Dict[str, float]:
        """Evaluate accuracy of list entities (precision, recall, F1)"""
        if not ground_truth and not predicted:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        if not ground_truth:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        if not predicted:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        # Convert to sets for comparison
        predicted_set = set(item.lower().strip() for item in predicted)
        ground_truth_set = set(item.lower().strip() for item in ground_truth)

        # Calculate metrics
        true_positives = len(predicted_set.intersection(ground_truth_set))
        precision = true_positives / len(predicted_set) if predicted_set else 0.0
        recall = true_positives / len(ground_truth_set) if ground_truth_set else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {"precision": precision, "recall": recall, "f1": f1}

    def evaluate_complex_entity(
        self, predicted: List[Dict], ground_truth: List[Dict], key_fields: List[str]
    ) -> Dict[str, float]:
        """Evaluate complex entities like education and experience"""
        if not ground_truth and not predicted:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        if not ground_truth:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        if not predicted:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        # For complex entities, we'll use a simplified approach
        # Compare the number of entries and key field matches
        predicted_count = len(predicted)
        ground_truth_count = len(ground_truth)

        # Count matches for key fields
        matches = 0
        total_fields = 0

        for pred_item in predicted:
            for gt_item in ground_truth:
                item_matches = 0
                item_total = 0
                for field in key_fields:
                    if field in pred_item and field in gt_item:
                        item_total += 1
                        if (
                            self.evaluate_entity_accuracy(
                                pred_item[field], gt_item[field]
                            )
                            > 0.8
                        ):
                            item_matches += 1

                if item_total > 0:
                    matches += item_matches / item_total
                    total_fields += 1

        precision = matches / predicted_count if predicted_count > 0 else 0.0
        recall = matches / ground_truth_count if ground_truth_count > 0 else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {"precision": precision, "recall": recall, "f1": f1}

    def evaluate_output(self, output_file: str) -> Dict[str, Any]:
        """Evaluate a single output file against ground truth"""
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                output_data = json.load(f)

            results = {
                "file": output_file,
                "entity_scores": {},
                "overall_score": 0.0,
                "field_scores": {},
            }

            # Evaluate each field
            for field in ["name", "email", "phone"]:
                predicted = output_data.get(field, "")
                ground_truth = self.ground_truth.get(field, "")
                score = self.evaluate_entity_accuracy(predicted, ground_truth)
                results["entity_scores"][field] = score
                results["field_scores"][field] = score

            # Evaluate list fields
            for field in ["skills", "certifications", "languages"]:
                predicted = output_data.get(field, [])
                ground_truth = self.ground_truth.get(field, [])
                scores = self.evaluate_list_accuracy(predicted, ground_truth)
                results["entity_scores"][field] = scores
                results["field_scores"][field] = scores["f1"]

            # Evaluate complex entities
            education_scores = self.evaluate_complex_entity(
                output_data.get("education", []),
                self.ground_truth.get("education", []),
                ["degree", "institution", "graduation_year"],
            )
            results["entity_scores"]["education"] = education_scores
            results["field_scores"]["education"] = education_scores["f1"]

            experience_scores = self.evaluate_complex_entity(
                output_data.get("experience", []),
                self.ground_truth.get("experience", []),
                ["job_title", "company", "years_worked", "description"],
            )
            results["entity_scores"]["experience"] = experience_scores
            results["field_scores"]["experience"] = experience_scores["f1"]

            # Calculate overall score (average of all field scores)
            field_scores = list(results["field_scores"].values())
            results["overall_score"] = (
                sum(field_scores) / len(field_scores) if field_scores else 0.0
            )

            return results

        except Exception as e:
            logger.error(f"Error evaluating {output_file}: {e}")
            return {"file": output_file, "error": str(e), "overall_score": 0.0}

    def evaluate_directory(self, output_dir: str = "../output") -> Dict[str, Any]:
        """Evaluate all JSON files in the output directory"""
        output_path = Path(output_dir)
        if not output_path.exists():
            logger.error(f"Output directory not found: {output_path}")
            return {}

        json_files = list(output_path.glob("*.json"))
        if not json_files:
            logger.warning(f"No JSON files found in {output_path}")
            return {}

        logger.info(f"Evaluating {len(json_files)} output files...")

        results = {
            "total_files": len(json_files),
            "evaluated_files": 0,
            "failed_files": 0,
            "file_results": {},
            "summary": {
                "overall_accuracy": 0.0,
                "field_accuracies": defaultdict(list),
                "best_file": "",
                "worst_file": "",
            },
        }

        best_score = 0.0
        worst_score = 1.0

        for json_file in json_files:
            file_result = self.evaluate_output(str(json_file))
            results["file_results"][json_file.name] = file_result

            if "error" not in file_result:
                results["evaluated_files"] += 1
                overall_score = file_result["overall_score"]

                # Track best and worst files
                if overall_score > best_score:
                    best_score = overall_score
                    results["summary"]["best_file"] = json_file.name

                if overall_score < worst_score:
                    worst_score = overall_score
                    results["summary"]["worst_file"] = json_file.name

                # Collect field scores for averaging
                for field, score in file_result["field_scores"].items():
                    results["summary"]["field_accuracies"][field].append(score)
            else:
                results["failed_files"] += 1

        # Calculate summary statistics
        if results["evaluated_files"] > 0:
            # Overall accuracy
            all_scores = [
                r["overall_score"]
                for r in results["file_results"].values()
                if "error" not in r
            ]
            results["summary"]["overall_accuracy"] = sum(all_scores) / len(all_scores)

            # Field accuracies
            for field, scores in results["summary"]["field_accuracies"].items():
                if scores:
                    results["summary"]["field_accuracies"][field] = sum(scores) / len(
                        scores
                    )

        return results

    def print_evaluation_summary(self, results: Dict[str, Any]):
        """Print a formatted evaluation summary"""
        print("\n" + "=" * 80)
        print("📊 RESUME PARSER EVALUATION SUMMARY")
        print("=" * 80)

        print(f"📁 Total files evaluated: {results['total_files']}")
        print(f"✅ Successfully evaluated: {results['evaluated_files']}")
        print(f"❌ Failed evaluations: {results['failed_files']}")

        if results["evaluated_files"] > 0:
            print(
                f"\n🎯 Overall Accuracy: {results['summary']['overall_accuracy']:.2%}"
            )

            print("\n📈 Field Accuracies:")
            for field, accuracy in results["summary"]["field_accuracies"].items():
                print(f"  {field.capitalize()}: {accuracy:.2%}")

            print(f"\n🏆 Best performing file: {results['summary']['best_file']}")
            print(f"⚠️  Worst performing file: {results['summary']['worst_file']}")

            # Detailed results for each file
            print("\n📄 Detailed Results:")
            for filename, file_result in results["file_results"].items():
                if "error" not in file_result:
                    print(f"  {filename}: {file_result['overall_score']:.2%}")
                else:
                    print(f"  {filename}: ERROR - {file_result['error']}")

        print("=" * 80)


def main():
    """Main evaluation function"""
    print("🧪 Resume Parser Evaluation")
    print("=" * 60)

    # Initialize evaluator
    evaluator = ResumeEvaluator()

    if not evaluator.ground_truth:
        print("❌ Failed to load ground truth data")
        return False

    print(f"✅ Ground truth loaded with {len(evaluator.ground_truth)} fields")

    # Evaluate output directory
    results = evaluator.evaluate_directory()

    if not results:
        print("❌ No evaluation results generated")
        return False

    # Print summary
    evaluator.print_evaluation_summary(results)

        # Save detailed results
    output_file = "../results/evaluation_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Detailed results saved to: {output_file}")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
