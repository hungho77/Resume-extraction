#!/usr/bin/env python3
"""
Comprehensive Resume Parser Evaluation
Advanced evaluation with detailed analysis, confidence intervals, and performance metrics
"""

import sys
import json
import time
from typing import Dict, List, Any
from pathlib import Path
import logging
from fuzzywuzzy import fuzz
from collections import defaultdict
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComprehensiveEvaluator:
    """Comprehensive resume parser evaluator with advanced metrics"""

    def __init__(self, ground_truth_file: str = "../results/ground_truth.json"):
        self.ground_truth_file = Path(ground_truth_file)
        self.ground_truth = self._load_ground_truth()
        self.evaluation_history = []

    def _load_ground_truth(self) -> Dict[str, Any]:
        """Load ground truth from JSON file"""
        if not self.ground_truth_file.exists():
            logger.error(f"Ground truth file not found: {self.ground_truth_file}")
            return {}

        try:
            with open(self.ground_truth_file, "r", encoding="utf-8") as f:
                ground_truth = json.load(f)
            logger.info(f"Loaded ground truth with {len(ground_truth)} fields")
            return ground_truth

        except Exception as e:
            logger.error(f"Error loading ground truth: {e}")
            return {}

    def evaluate_entity_accuracy(
        self, predicted: str, ground_truth: str
    ) -> Dict[str, float]:
        """Evaluate accuracy of string entities with detailed metrics"""
        if not ground_truth and not predicted:
            return {
                "exact_match": 1.0,
                "fuzzy_match": 1.0,
                "partial_match": 1.0,
                "score": 1.0,
            }
        if not ground_truth or not predicted:
            return {
                "exact_match": 0.0,
                "fuzzy_match": 0.0,
                "partial_match": 0.0,
                "score": 0.0,
            }

        # Multiple matching strategies
        exact_match = 1.0 if predicted.lower() == ground_truth.lower() else 0.0
        fuzzy_match = fuzz.ratio(predicted.lower(), ground_truth.lower()) / 100.0
        partial_match = (
            fuzz.partial_ratio(predicted.lower(), ground_truth.lower()) / 100.0
        )

        # Combined score (weighted average)
        score = exact_match * 0.4 + fuzzy_match * 0.4 + partial_match * 0.2

        return {
            "exact_match": exact_match,
            "fuzzy_match": fuzzy_match,
            "partial_match": partial_match,
            "score": score,
        }

    def evaluate_list_accuracy(
        self, predicted: List[str], ground_truth: List[str]
    ) -> Dict[str, float]:
        """Evaluate accuracy of list entities with detailed metrics"""
        if not ground_truth and not predicted:
            return {
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "exact_match": 1.0,
                "partial_match": 1.0,
            }
        if not ground_truth:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "exact_match": 0.0,
                "partial_match": 0.0,
            }
        if not predicted:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "exact_match": 0.0,
                "partial_match": 0.0,
            }

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

        # Exact match (all items match exactly)
        exact_match = 1.0 if predicted_set == ground_truth_set else 0.0

        # Partial match (some items match)
        partial_match = (
            len(predicted_set.intersection(ground_truth_set)) / len(ground_truth_set)
            if ground_truth_set
            else 0.0
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "exact_match": exact_match,
            "partial_match": partial_match,
        }

    def evaluate_complex_entity(
        self, predicted: List[Dict], ground_truth: List[Dict], key_fields: List[str]
    ) -> Dict[str, float]:
        """Evaluate complex entities with detailed metrics"""
        if not ground_truth and not predicted:
            return {
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "field_accuracy": 1.0,
                "structure_accuracy": 1.0,
            }
        if not ground_truth:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "field_accuracy": 0.0,
                "structure_accuracy": 0.0,
            }
        if not predicted:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "field_accuracy": 0.0,
                "structure_accuracy": 0.0,
            }

        # Structure accuracy (number of entries)
        structure_accuracy = min(len(predicted), len(ground_truth)) / max(
            len(predicted), len(ground_truth)
        )

        # Field-level accuracy
        total_field_matches = 0
        total_fields = 0

        for pred_item in predicted:
            for gt_item in ground_truth:
                item_field_matches = 0
                item_total_fields = 0

                for field in key_fields:
                    if field in pred_item and field in gt_item:
                        item_total_fields += 1
                        field_accuracy = self.evaluate_entity_accuracy(
                            pred_item[field], gt_item[field]
                        )
                        if field_accuracy["score"] > 0.8:
                            item_field_matches += 1

                if item_total_fields > 0:
                    total_field_matches += item_field_matches / item_total_fields
                    total_fields += 1

        field_accuracy = total_field_matches / total_fields if total_fields > 0 else 0.0

        # Overall metrics
        precision = field_accuracy * structure_accuracy
        recall = field_accuracy * structure_accuracy
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "field_accuracy": field_accuracy,
            "structure_accuracy": structure_accuracy,
        }

    def evaluate_output(self, output_file: str) -> Dict[str, Any]:
        """Evaluate a single output file with comprehensive metrics"""
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                output_data = json.load(f)

            results = {
                "file": output_file,
                "evaluation_timestamp": time.time(),
                "entity_scores": {},
                "field_scores": {},
                "overall_score": 0.0,
                "confidence_metrics": {},
                "error_analysis": {},
            }

            # Evaluate string fields
            for field in ["name", "email", "phone"]:
                predicted = output_data.get(field, "")
                ground_truth = self.ground_truth.get(field, "")
                scores = self.evaluate_entity_accuracy(predicted, ground_truth)
                results["entity_scores"][field] = scores
                results["field_scores"][field] = scores["score"]

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

            # Calculate overall score
            field_scores = list(results["field_scores"].values())
            results["overall_score"] = (
                sum(field_scores) / len(field_scores) if field_scores else 0.0
            )

            # Calculate confidence metrics
            results["confidence_metrics"] = {
                "score_variance": statistics.variance(field_scores)
                if len(field_scores) > 1
                else 0.0,
                "score_std": statistics.stdev(field_scores)
                if len(field_scores) > 1
                else 0.0,
                "min_score": min(field_scores) if field_scores else 0.0,
                "max_score": max(field_scores) if field_scores else 0.0,
                "median_score": statistics.median(field_scores)
                if field_scores
                else 0.0,
            }

            # Error analysis
            results["error_analysis"] = {
                "missing_fields": [
                    field
                    for field, score in results["field_scores"].items()
                    if score == 0.0
                ],
                "low_performing_fields": [
                    field
                    for field, score in results["field_scores"].items()
                    if score < 0.5
                ],
                "high_performing_fields": [
                    field
                    for field, score in results["field_scores"].items()
                    if score > 0.8
                ],
            }

            return results

        except Exception as e:
            logger.error(f"Error evaluating {output_file}: {e}")
            return {"file": output_file, "error": str(e), "overall_score": 0.0}

    def evaluate_directory(self, output_dir: str = "../output") -> Dict[str, Any]:
        """Evaluate all JSON files in the output directory with comprehensive analysis"""
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
            "evaluation_timestamp": time.time(),
            "total_files": len(json_files),
            "evaluated_files": 0,
            "failed_files": 0,
            "file_results": {},
            "summary": {
                "overall_accuracy": 0.0,
                "field_accuracies": defaultdict(list),
                "confidence_intervals": {},
                "performance_analysis": {},
                "best_file": "",
                "worst_file": "",
                "recommendations": [],
            },
        }

        all_scores = []
        field_scores_by_file = defaultdict(list)

        for json_file in json_files:
            file_result = self.evaluate_output(str(json_file))
            results["file_results"][json_file.name] = file_result

            if "error" not in file_result:
                results["evaluated_files"] += 1
                overall_score = file_result["overall_score"]
                all_scores.append(overall_score)

                # Collect field scores for analysis
                for field, score in file_result["field_scores"].items():
                    field_scores_by_file[field].append(score)

                # Track best and worst files
                if not results["summary"]["best_file"] or overall_score > max(
                    [
                        r["overall_score"]
                        for r in results["file_results"].values()
                        if "error" not in r
                    ]
                ):
                    results["summary"]["best_file"] = json_file.name

                if not results["summary"]["worst_file"] or overall_score < min(
                    [
                        r["overall_score"]
                        for r in results["file_results"].values()
                        if "error" not in r
                    ]
                ):
                    results["summary"]["worst_file"] = json_file.name
            else:
                results["failed_files"] += 1

        # Calculate comprehensive summary statistics
        if results["evaluated_files"] > 0:
            # Overall accuracy
            results["summary"]["overall_accuracy"] = sum(all_scores) / len(all_scores)

            # Field accuracies with confidence intervals
            for field, scores in field_scores_by_file.items():
                if scores:
                    mean_score = sum(scores) / len(scores)
                    results["summary"]["field_accuracies"][field] = mean_score

                    # Calculate confidence interval (95%)
                    if len(scores) > 1:
                        std_error = statistics.stdev(scores) / (len(scores) ** 0.5)
                        confidence_interval = 1.96 * std_error  # 95% CI
                        results["summary"]["confidence_intervals"][field] = {
                            "mean": mean_score,
                            "std_error": std_error,
                            "confidence_interval": confidence_interval,
                            "lower_bound": max(0, mean_score - confidence_interval),
                            "upper_bound": min(1, mean_score + confidence_interval),
                        }

            # Performance analysis
            results["summary"]["performance_analysis"] = {
                "score_distribution": {
                    "mean": statistics.mean(all_scores) if all_scores else 0.0,
                    "median": statistics.median(all_scores) if all_scores else 0.0,
                    "std": statistics.stdev(all_scores) if len(all_scores) > 1 else 0.0,
                    "min": min(all_scores) if all_scores else 0.0,
                    "max": max(all_scores) if all_scores else 0.0,
                },
                "performance_tiers": {
                    "excellent": len([s for s in all_scores if s >= 0.8]),
                    "good": len([s for s in all_scores if 0.6 <= s < 0.8]),
                    "fair": len([s for s in all_scores if 0.4 <= s < 0.6]),
                    "poor": len([s for s in all_scores if s < 0.4]),
                },
            }

            # Generate recommendations
            results["summary"]["recommendations"] = self._generate_recommendations(
                results
            )

        return results

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations based on evaluation results"""
        recommendations = []

        if results["evaluated_files"] == 0:
            recommendations.append(
                "No files were successfully evaluated. Check input format and ground truth."
            )
            return recommendations

        overall_accuracy = results["summary"]["overall_accuracy"]

        if overall_accuracy < 0.5:
            recommendations.append(
                "Overall accuracy is below 50%. Consider improving the extraction model."
            )

        if overall_accuracy < 0.7:
            recommendations.append(
                "Overall accuracy is below 70%. Review field-specific extraction logic."
            )

        # Field-specific recommendations
        field_accuracies = results["summary"]["field_accuracies"]

        for field, accuracy in field_accuracies.items():
            if accuracy < 0.3:
                recommendations.append(
                    f"Field '{field}' has very low accuracy ({accuracy:.1%}). Consider specialized extraction for this field."
                )
            elif accuracy < 0.6:
                recommendations.append(
                    f"Field '{field}' has moderate accuracy ({accuracy:.1%}). Review extraction patterns for this field."
                )

        # Performance tier recommendations
        performance_tiers = results["summary"]["performance_analysis"][
            "performance_tiers"
        ]

        if performance_tiers["poor"] > 0:
            recommendations.append(
                f"{performance_tiers['poor']} file(s) performed poorly. Review these specific cases."
            )

        if performance_tiers["excellent"] == 0:
            recommendations.append(
                "No files achieved excellent performance. Consider model improvements."
            )

        return recommendations

    def print_comprehensive_summary(self, results: Dict[str, Any]):
        """Print a comprehensive evaluation summary"""
        print("\n" + "=" * 100)
        print("📊 COMPREHENSIVE RESUME PARSER EVALUATION SUMMARY")
        print("=" * 100)

        print(f"📁 Total files evaluated: {results['total_files']}")
        print(f"✅ Successfully evaluated: {results['evaluated_files']}")
        print(f"❌ Failed evaluations: {results['failed_files']}")

        if results["evaluated_files"] > 0:
            print(
                f"\n🎯 Overall Accuracy: {results['summary']['overall_accuracy']:.2%}"
            )

            # Performance analysis
            perf_analysis = results["summary"]["performance_analysis"]
            print("\n📈 Performance Analysis:")
            print(f"  Mean Score: {perf_analysis['score_distribution']['mean']:.3f}")
            print(
                f"  Median Score: {perf_analysis['score_distribution']['median']:.3f}"
            )
            print(
                f"  Standard Deviation: {perf_analysis['score_distribution']['std']:.3f}"
            )
            print(
                f"  Score Range: {perf_analysis['score_distribution']['min']:.3f} - {perf_analysis['score_distribution']['max']:.3f}"
            )

            # Performance tiers
            tiers = perf_analysis["performance_tiers"]
            print("\n🏆 Performance Tiers:")
            print(f"  Excellent (≥80%): {tiers['excellent']} files")
            print(f"  Good (60-79%): {tiers['good']} files")
            print(f"  Fair (40-59%): {tiers['fair']} files")
            print(f"  Poor (<40%): {tiers['poor']} files")

            # Field accuracies with confidence intervals
            print("\n📊 Field Accuracies:")
            for field, accuracy in results["summary"]["field_accuracies"].items():
                if field in results["summary"]["confidence_intervals"]:
                    ci = results["summary"]["confidence_intervals"][field]
                    print(
                        f"  {field.capitalize()}: {accuracy:.2%} (95% CI: {ci['lower_bound']:.2%} - {ci['upper_bound']:.2%})"
                    )
                else:
                    print(f"  {field.capitalize()}: {accuracy:.2%}")

            print(f"\n🏆 Best performing file: {results['summary']['best_file']}")
            print(f"⚠️  Worst performing file: {results['summary']['worst_file']}")

            # Recommendations
            if results["summary"]["recommendations"]:
                print("\n💡 Recommendations:")
                for i, rec in enumerate(results["summary"]["recommendations"], 1):
                    print(f"  {i}. {rec}")

            # Detailed results
            print("\n📄 Detailed Results:")
            for filename, file_result in results["file_results"].items():
                if "error" not in file_result:
                    print(f"  {filename}: {file_result['overall_score']:.2%}")
                else:
                    print(f"  {filename}: ERROR - {file_result['error']}")

        print("=" * 100)


def main():
    """Main comprehensive evaluation function"""
    print("🧪 Comprehensive Resume Parser Evaluation")
    print("=" * 80)

    # Initialize evaluator
    evaluator = ComprehensiveEvaluator()

    if not evaluator.ground_truth:
        print("❌ Failed to load ground truth data")
        return False

    print(f"✅ Ground truth loaded with {len(evaluator.ground_truth)} fields")

    # Evaluate output directory
    results = evaluator.evaluate_directory()

    if not results:
        print("❌ No evaluation results generated")
        return False

    # Print comprehensive summary
    evaluator.print_comprehensive_summary(results)

    # Save detailed results
    output_file = "../results/comprehensive_evaluation_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Detailed results saved to: {output_file}")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
