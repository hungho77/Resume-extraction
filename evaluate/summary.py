#!/usr/bin/env python3
"""
Evaluation Summary Script
Provides a quick overview of resume parser evaluation results
"""

import json


def load_evaluation_results(
    file_path: str = "../results/comprehensive_evaluation_results.json",
):
    """Load evaluation results from JSON file"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Evaluation results file not found: {file_path}")
        return None


def print_summary(results):
    """Print a concise summary of evaluation results"""
    print("📊 RESUME PARSER EVALUATION SUMMARY")
    print("=" * 60)

    print(f"📁 Files Evaluated: {results['evaluated_files']}/{results['total_files']}")
    print(f"🎯 Overall Accuracy: {results['summary']['overall_accuracy']:.1%}")

    # Performance tiers
    tiers = results["summary"]["performance_analysis"]["performance_tiers"]
    print("\n🏆 Performance Breakdown:")
    print(f"  Excellent (≥80%): {tiers['excellent']} files")
    print(f"  Good (60-79%): {tiers['good']} files")
    print(f"  Fair (40-59%): {tiers['fair']} files")
    print(f"  Poor (<40%): {tiers['poor']} files")

    # Field performance
    print("\n📈 Field Performance:")
    field_accuracies = results["summary"]["field_accuracies"]
    for field, accuracy in field_accuracies.items():
        status = "✅" if accuracy >= 0.7 else "⚠️" if accuracy >= 0.4 else "❌"
        print(f"  {status} {field.capitalize()}: {accuracy:.1%}")

    # Key findings
    print("\n🔍 Key Findings:")
    if results["summary"]["overall_accuracy"] < 0.5:
        print("  • Overall accuracy below 50% - significant improvements needed")
    elif results["summary"]["overall_accuracy"] < 0.7:
        print("  • Overall accuracy below 70% - moderate improvements needed")
    else:
        print("  • Overall accuracy above 70% - good performance")

    # Best and worst performers
    print(f"  • Best file: {results['summary']['best_file']}")
    print(f"  • Worst file: {results['summary']['worst_file']}")

    # Recommendations count
    rec_count = len(results["summary"]["recommendations"])
    print(f"  • {rec_count} improvement recommendations generated")

    print("=" * 60)


def main():
    """Main summary function"""
    print("📋 Resume Parser Evaluation Summary")
    print("=" * 60)

    # Load results
    results = load_evaluation_results()
    if not results:
        return False

    # Print summary
    print_summary(results)

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
