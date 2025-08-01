#!/usr/bin/env python3
"""
Results Management Script
Manages and organizes all evaluation results in the results/ folder
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any


class ResultsManager:
    """Manages evaluation results and analysis files"""

    def __init__(self, results_dir: str = "../results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

    def save_ground_truth(
        self, ground_truth: Dict[str, Any], filename: str = "ground_truth.json"
    ):
        """Save ground truth to results folder"""
        filepath = self.results_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(ground_truth, f, indent=2, ensure_ascii=False)
        print(f"✅ Ground truth saved to: {filepath}")
        return filepath

    def save_evaluation_results(
        self, results: Dict[str, Any], filename: str = "evaluation_results.json"
    ):
        """Save evaluation results to results folder"""
        filepath = self.results_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✅ Evaluation results saved to: {filepath}")
        return filepath

    def save_comprehensive_results(
        self,
        results: Dict[str, Any],
        filename: str = "comprehensive_evaluation_results.json",
    ):
        """Save comprehensive evaluation results to results folder"""
        filepath = self.results_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✅ Comprehensive results saved to: {filepath}")
        return filepath

    def save_analysis_summary(
        self, summary: Dict[str, Any], filename: str = "analysis_summary.json"
    ):
        """Save analysis summary to results folder"""
        filepath = self.results_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"✅ Analysis summary saved to: {filepath}")
        return filepath

    def load_results(self, filename: str) -> Dict[str, Any]:
        """Load results from results folder"""
        filepath = self.results_dir / filename
        if not filepath.exists():
            print(f"❌ Results file not found: {filepath}")
            return {}

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading results: {e}")
            return {}

    def list_results_files(self) -> List[str]:
        """List all results files in the results folder"""
        files = []
        for file in self.results_dir.glob("*.json"):
            files.append(file.name)
        return sorted(files)

    def create_results_index(self) -> Dict[str, Any]:
        """Create an index of all results files with metadata"""
        index = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results_directory": str(self.results_dir),
            "files": {},
        }

        for file in self.results_dir.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Extract key information based on file type
                file_info = {
                    "size_bytes": file.stat().st_size,
                    "last_modified": time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(file.stat().st_mtime)
                    ),
                }

                if "ground_truth" in file.name:
                    file_info["type"] = "ground_truth"
                    file_info["fields"] = len(data) if isinstance(data, dict) else 0
                elif "comprehensive" in file.name:
                    file_info["type"] = "comprehensive_evaluation"
                    if "summary" in data:
                        file_info["overall_accuracy"] = data["summary"].get(
                            "overall_accuracy", 0
                        )
                        file_info["files_evaluated"] = data.get("evaluated_files", 0)
                elif "evaluation" in file.name:
                    file_info["type"] = "basic_evaluation"
                    if "summary" in data:
                        file_info["overall_accuracy"] = data["summary"].get(
                            "overall_accuracy", 0
                        )
                        file_info["files_evaluated"] = data.get("evaluated_files", 0)
                else:
                    file_info["type"] = "other"

                index["files"][file.name] = file_info

            except Exception as e:
                print(f"⚠️ Error processing {file.name}: {e}")

        return index

    def save_results_index(
        self, index: Dict[str, Any], filename: str = "results_index.json"
    ):
        """Save results index to results folder"""
        filepath = self.results_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, ensure_ascii=False)
        print(f"✅ Results index saved to: {filepath}")
        return filepath

    def print_results_overview(self):
        """Print an overview of all results files"""
        print("\n" + "=" * 80)
        print("📁 RESULTS FOLDER OVERVIEW")
        print("=" * 80)

        files = self.list_results_files()
        if not files:
            print("❌ No results files found in results/ folder")
            return

        print(f"📂 Results directory: {self.results_dir}")
        print(f"📄 Total files: {len(files)}")

        # Create and display index
        index = self.create_results_index()

        print("\n📊 Results Files:")
        for filename, info in index["files"].items():
            file_type = info.get("type", "unknown")
            size_mb = info["size_bytes"] / (1024 * 1024)

            print(f"  📄 {filename}")
            print(f"     Type: {file_type}")
            print(f"     Size: {size_mb:.2f} MB")
            print(f"     Modified: {info['last_modified']}")

            if "overall_accuracy" in info:
                print(f"     Overall Accuracy: {info['overall_accuracy']:.1%}")
            if "files_evaluated" in info:
                print(f"     Files Evaluated: {info['files_evaluated']}")
            if "fields" in info:
                print(f"     Fields: {info['fields']}")
            print()

        # Save index
        self.save_results_index(index)

        print("=" * 80)

    def cleanup_old_results(self, days_old: int = 30):
        """Clean up old results files"""
        import time

        current_time = time.time()
        cutoff_time = current_time - (days_old * 24 * 60 * 60)

        cleaned_files = []
        for file in self.results_dir.glob("*.json"):
            if file.stat().st_mtime < cutoff_time:
                try:
                    file.unlink()
                    cleaned_files.append(file.name)
                except Exception as e:
                    print(f"❌ Error deleting {file.name}: {e}")

        if cleaned_files:
            print(f"🧹 Cleaned up {len(cleaned_files)} old files:")
            for filename in cleaned_files:
                print(f"  - {filename}")
        else:
            print("✅ No old files to clean up")


def main():
    """Main results management function"""
    print("📁 Results Management")
    print("=" * 60)

    manager = ResultsManager()

    # Print overview
    manager.print_results_overview()

    # Optionally clean up old files (uncomment if needed)
    # manager.cleanup_old_results(days_old=30)

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
