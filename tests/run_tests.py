#!/usr/bin/env python3
"""
Test Runner for Resume Parser
Runs all test files and provides a summary of results.
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path
from typing import Dict, List

# Add parent directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestRunner:
    """Test runner for the resume parser project"""

    def __init__(self):
        self.test_files = self._discover_test_files()
        self.results = {}
        self.start_time = None

    def _discover_test_files(self) -> List[str]:
        """Discover all test files in the tests directory"""
        test_dir = Path(__file__).parent
        test_files = []

        for test_file in test_dir.glob("test_*.py"):
            if test_file.name != "run_tests.py":
                test_files.append(str(test_file))

        return sorted(test_files)

    def run_test_file(self, test_file: str) -> Dict:
        """Run a single test file and return results"""
        print(f"\n🧪 Running {Path(test_file).name}...")
        print("=" * 50)

        start_time = time.time()

        try:
            # Run the test file
            result = subprocess.run(
                [sys.executable, test_file],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent,  # Run from project root
            )

            end_time = time.time()
            duration = end_time - start_time

            if result.returncode == 0:
                print(
                    f"✅ {Path(test_file).name} completed successfully ({duration:.2f}s)"
                )
                return {
                    "file": Path(test_file).name,
                    "status": "PASSED",
                    "duration": duration,
                    "output": result.stdout,
                    "error": result.stderr,
                }
            else:
                print(f"❌ {Path(test_file).name} failed ({duration:.2f}s)")
                if result.stderr:
                    print(f"Error: {result.stderr}")
                return {
                    "file": Path(test_file).name,
                    "status": "FAILED",
                    "duration": duration,
                    "output": result.stdout,
                    "error": result.stderr,
                }

        except Exception as e:
            print(f"❌ {Path(test_file).name} crashed: {e}")
            return {
                "file": Path(test_file).name,
                "status": "CRASHED",
                "duration": 0,
                "output": "",
                "error": str(e),
            }

    def run_all_tests(self) -> bool:
        """Run all discovered test files"""
        print("🚀 Resume Parser Test Suite")
        print("=" * 50)
        print(f"📁 Found {len(self.test_files)} test files")

        self.start_time = time.time()
        total_passed = 0
        total_failed = 0

        for test_file in self.test_files:
            result = self.run_test_file(test_file)
            self.results[result["file"]] = result

            if result["status"] == "PASSED":
                total_passed += 1
            else:
                total_failed += 1

        # Print summary
        self._print_summary(total_passed, total_failed)

        return total_failed == 0

    def run_specific_test(self, test_name: str) -> bool:
        """Run a specific test file"""
        test_file = None

        # Try different variations of the test name
        possible_names = [
            f"tests/{test_name}.py",
            f"tests/test_{test_name}.py",
            f"{test_name}.py",
            f"test_{test_name}.py",
        ]

        for name in possible_names:
            if os.path.exists(name):
                test_file = name
                break

        if not test_file:
            print(f"❌ Test file not found for: {test_name}")
            print("Available tests:")
            for tf in self.test_files:
                print(f"  - {Path(tf).name}")
            return False

        result = self.run_test_file(test_file)
        self.results[result["file"]] = result

        if result["status"] == "PASSED":
            print(f"\n✅ {Path(test_file).name} passed!")
            return True
        else:
            print(f"\n❌ {Path(test_file).name} failed!")
            if result["error"]:
                print(f"Error: {result['error']}")
            return False

    def _print_summary(self, passed: int, failed: int):
        """Print test summary"""
        total_time = time.time() - self.start_time if self.start_time else 0

        print("\n" + "=" * 50)
        print("📊 Test Summary")
        print("=" * 50)

        for filename, result in self.results.items():
            status_icon = "✅" if result["status"] == "PASSED" else "❌"
            print(
                f"{status_icon} {filename}: {result['status']} ({result['duration']:.2f}s)"
            )

        print("\n📈 Overall Results:")
        print(f"   ✅ Passed: {passed}")
        print(f"   ❌ Failed: {failed}")
        print(f"   📊 Total: {len(self.results)}")
        print(f"   ⏱️  Total Time: {total_time:.2f}s")

        if failed == 0:
            print("\n🎉 All tests passed!")
        else:
            print(f"\n⚠️  {failed} test(s) failed")

    def save_results(self, filename: str = "test_results.json"):
        """Save test results to JSON file"""
        output_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_time": time.time() - self.start_time if self.start_time else 0,
            "results": self.results,
        }

        try:
            with open(filename, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"💾 Results saved to: {filename}")
        except Exception as e:
            print(f"❌ Failed to save results: {e}")

    def show_help(self):
        """Show help information"""
        print("🚀 Resume Parser Test Runner")
        print("=" * 30)
        print("Usage:")
        print("  python run_tests.py              # Run all tests")
        print("  python run_tests.py <test_name>  # Run specific test")
        print("  python run_tests.py help         # Show this help")
        print("\nAvailable tests:")
        for test_file in self.test_files:
            print(f"  - {Path(test_file).name}")
        print("\nExamples:")
        print("  python run_tests.py pdf_resume_extraction")
        print("  python run_tests.py llm_client")
        print("  python run_tests.py setup")


def main():
    """Main function"""
    runner = TestRunner()

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "help":
            runner.show_help()
        else:
            # Run specific test
            success = runner.run_specific_test(command)
            runner.save_results()
            sys.exit(0 if success else 1)
    else:
        # Run all tests
        success = runner.run_all_tests()
        runner.save_results()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
