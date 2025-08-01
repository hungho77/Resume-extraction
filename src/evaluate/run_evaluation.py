#!/usr/bin/env python3
"""
Comprehensive Evaluation Runner
Runs evaluation pipeline with configurable paths for outputs and CSV data
"""

import sys
import subprocess
import argparse
import json
import csv
from pathlib import Path
from datetime import datetime


def run_script(script_name: str, description: str) -> bool:
    """Run a Python script and return success status"""
    print(f"\n🔄 {description}")
    print("-" * 50)

    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
        )

        if result.returncode == 0:
            print("✅ Script completed successfully")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            print("❌ Script failed")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Error running {script_name}: {e}")
        return False


def create_ground_truth_from_csv(csv_path: str):
    """Create ground truth files from CSV"""
    print("🔧 Creating Ground Truth from CSV")
    print("=" * 60)
    
    # Create GT directory
    gt_dir = Path("../../data/GT")
    gt_dir.mkdir(parents=True, exist_ok=True)
    
    # Create results directory
    results_dir = Path("../../results")
    results_dir.mkdir(exist_ok=True)
    
    csv_file = Path(csv_path)
    if not csv_file.exists():
        print(f"❌ CSV file not found: {csv_file}")
        return False
    
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
                
                # Parse the resume text using the existing function
                import sys
                import os
                # Add the evaluate directory to the path
                evaluate_dir = os.path.dirname(os.path.abspath(__file__))
                sys.path.insert(0, evaluate_dir)
                from create_gt import parse_resume_text
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
        return False
    
    # Save main ground truth file
    main_gt_file = results_dir / "ground_truth.json"
    with open(main_gt_file, "w", encoding="utf-8") as f:
        json.dump(all_ground_truth, f, indent=2, ensure_ascii=False)
    
    print("\n📊 Summary:")
    print(f"  📄 Total files processed: {created_files}")
    print(f"  ✅ Ground truth files created: {created_files}")
    print(f"  📂 Individual files location: {gt_dir.absolute()}")
    print(f"  📂 Main ground truth: {main_gt_file}")
    
    return True


def evaluate_outputs_against_ground_truth(outputs_dir: str):
    """Evaluate JSON files in outputs directory against ground truth"""
    print("🔧 Evaluating Outputs Against Ground Truth")
    print("=" * 60)
    
    outputs_path = Path(outputs_dir)
    gt_dir = Path("../../data/GT")
    results_dir = Path("../../results")
    
    if not outputs_path.exists():
        print(f"❌ Outputs directory not found: {outputs_path}")
        return False
    
    if not gt_dir.exists():
        print(f"❌ Ground truth directory not found: {gt_dir}")
        return False
    
    # Get all JSON files in outputs
    output_files = list(outputs_path.glob("*.json"))
    if not output_files:
        print("❌ No JSON files found in outputs directory")
        return False
    
    print(f"📁 Found {len(output_files)} output files")
    
    evaluation_results = {
        "total_files": len(output_files),
        "successful_evaluations": 0,
        "failed_evaluations": 0,
        "results": {},
        "summary": {
            "overall_accuracy": 0.0,
            "field_accuracies": {},
            "best_file": "",
            "worst_file": "",
            "average_score": 0.0
        }
    }
    
    total_scores = []
    field_scores = {}
    
    for output_file in output_files:
        # Extract ID from filename (remove .json extension)
        file_id = output_file.stem
        
        # Look for corresponding ground truth file
        gt_file = gt_dir / f"{file_id}_gt.json"
        
        if not gt_file.exists():
            print(f"⚠️ No ground truth found for {file_id}")
            evaluation_results["failed_evaluations"] += 1
            continue
        
        try:
            # Load output and ground truth
            with open(output_file, "r", encoding="utf-8") as f:
                output_data = json.load(f)
            
            with open(gt_file, "r", encoding="utf-8") as f:
                ground_truth = json.load(f)
            
            # Evaluate using the existing evaluation logic
            from evaluate import ResumeEvaluator
            evaluator = ResumeEvaluator(str(gt_file))
            
            # Pass ground truth data to evaluator for comparison
            evaluator.ground_truth = ground_truth
            
            # Create a temporary file for the output
            temp_output_file = Path("../temp_output.json")
            with open(temp_output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            # Evaluate the file
            result = evaluator.evaluate_output(str(temp_output_file))
            
            # Clean up temp file
            temp_output_file.unlink()
            
            if result:
                evaluation_results["successful_evaluations"] += 1
                evaluation_results["results"][file_id] = result
                
                # Track scores
                overall_score = result.get("overall_accuracy", 0.0)
                total_scores.append(overall_score)
                
                # Track field scores
                field_accuracies = result.get("field_accuracies", {})
                for field, score in field_accuracies.items():
                    if field not in field_scores:
                        field_scores[field] = []
                    field_scores[field].append(score)
                
                print(f"✅ Evaluated {file_id}: {overall_score:.2f}%")
            else:
                evaluation_results["failed_evaluations"] += 1
                print(f"❌ Failed to evaluate {file_id}")
        
        except Exception as e:
            print(f"❌ Error evaluating {file_id}: {e}")
            evaluation_results["failed_evaluations"] += 1
    
    # Calculate summary statistics
    if total_scores:
        evaluation_results["summary"]["average_score"] = sum(total_scores) / len(total_scores)
        evaluation_results["summary"]["overall_accuracy"] = evaluation_results["summary"]["average_score"]
        
        # Find best and worst files
        best_score = max(total_scores)
        worst_score = min(total_scores)
        
        for file_id, result in evaluation_results["results"].items():
            if result.get("overall_accuracy", 0) == best_score:
                evaluation_results["summary"]["best_file"] = file_id
            if result.get("overall_accuracy", 0) == worst_score:
                evaluation_results["summary"]["worst_file"] = file_id
        
        # Calculate field accuracies
        for field, scores in field_scores.items():
            evaluation_results["summary"]["field_accuracies"][field] = sum(scores) / len(scores)
    
    # Save evaluation results
    eval_file = results_dir / "custom_evaluation_results.json"
    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n📊 EVALUATION SUMMARY")
    print("=" * 60)
    print(f"📁 Total files: {evaluation_results['total_files']}")
    print(f"✅ Successful: {evaluation_results['successful_evaluations']}")
    print(f"❌ Failed: {evaluation_results['failed_evaluations']}")
    print(f"🎯 Overall Accuracy: {evaluation_results['summary']['overall_accuracy']:.2f}%")
    
    if evaluation_results["summary"]["field_accuracies"]:
        print("\n📈 Field Accuracies:")
        for field, accuracy in evaluation_results["summary"]["field_accuracies"].items():
            print(f"  {field}: {accuracy:.2f}%")
    
    print(f"\n🏆 Best file: {evaluation_results['summary']['best_file']}")
    print(f"⚠️  Worst file: {evaluation_results['summary']['worst_file']}")
    print(f"💾 Results saved to: {eval_file}")
    
    return True


def run_standard_evaluation():
    """Run the standard evaluation pipeline"""
    print("🔧 Running Standard Evaluation Pipeline")
    print("=" * 60)
    
    steps = []
    
    # Step 1: Create ground truth from CSV
    print("\n" + "=" * 80)
    print("STEP 1: Creating Ground Truth from CSV")
    print("=" * 80)
    
    success = run_script(
        "create_gt.py", "Creating ground truth files from CSV data"
    )
    steps.append(("Create Ground Truth from CSV", success))
    
    if not success:
        print("❌ Ground truth creation failed. Stopping evaluation.")
        return False

    # Step 2: Run basic evaluation
    print("\n" + "=" * 80)
    print("STEP 2: Basic Evaluation")
    print("=" * 80)

    success = run_script("evaluate.py", "Running basic evaluation")
    steps.append(("Basic Evaluation", success))

    # Step 3: Run comprehensive evaluation
    print("\n" + "=" * 80)
    print("STEP 3: Comprehensive Evaluation")
    print("=" * 80)

    success = run_script("comprehensive_eval.py", "Running comprehensive evaluation")
    steps.append(("Comprehensive Evaluation", success))

    # Step 4: Generate summary
    print("\n" + "=" * 80)
    print("STEP 4: Generate Summary")
    print("=" * 80)

    success = run_script("summary.py", "Generating evaluation summary")
    steps.append(("Summary Generation", success))

    # Step 5: Results management
    print("\n" + "=" * 80)
    print("STEP 5: Results Management")
    print("=" * 80)

    success = run_script("manage_results.py", "Managing and organizing results")
    steps.append(("Results Management", success))
    
    return steps


def main():
    """Main evaluation pipeline with configurable paths"""
    parser = argparse.ArgumentParser(description="Comprehensive Evaluation Pipeline")
    parser.add_argument(
        "--csv-path", 
        type=str, 
        default="../data/Resume/Resume.csv",
        help="Path to CSV file with resume data (default: ../data/Resume/Resume.csv)"
    )
    parser.add_argument(
        "--outputs-dir", 
        type=str, 
        default="../outputs",
        help="Path to outputs directory with JSON files (default: ../outputs)"
    )
    parser.add_argument(
        "--mode", 
        choices=["standard", "custom"], 
        default="custom",
        help="Evaluation mode: standard (uses existing scripts) or custom (direct evaluation)"
    )
    
    args = parser.parse_args()
    
    print("🚀 COMPREHENSIVE EVALUATION PIPELINE")
    print("=" * 80)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Working directory: {Path.cwd()}")
    print(f"📄 CSV Path: {args.csv_path}")
    print(f"📂 Outputs Directory: {args.outputs_dir}")
    print(f"🔧 Mode: {args.mode}")

    # Create results directory
    results_dir = Path("../results")
    results_dir.mkdir(exist_ok=True)
    print(f"📁 Results directory: {results_dir.absolute()}")

    steps = []
    
    if args.mode == "standard":
        # Run standard evaluation pipeline
        steps = run_standard_evaluation()
    else:
        # Run custom evaluation pipeline
        print("\n" + "=" * 80)
        print("STEP 1: Creating Ground Truth from CSV")
        print("=" * 80)
        
        success = create_ground_truth_from_csv(args.csv_path)
        steps.append(("Create Ground Truth from CSV", success))
        
        if not success:
            print("❌ Ground truth creation failed. Stopping evaluation.")
            return False
        
        # Step 2: Evaluate outputs against ground truth
        print("\n" + "=" * 80)
        print("STEP 2: Evaluating Outputs Against Ground Truth")
        print("=" * 80)
        
        success = evaluate_outputs_against_ground_truth(args.outputs_dir)
        steps.append(("Evaluate Outputs", success))

    # Print final summary
    print("\n" + "=" * 80)
    print("📊 EVALUATION PIPELINE SUMMARY")
    print("=" * 80)

    successful_steps = sum(1 for _, success in steps if success)
    total_steps = len(steps)

    print(f"✅ Successful steps: {successful_steps}/{total_steps}")
    print(f"❌ Failed steps: {total_steps - successful_steps}/{total_steps}")

    print("\n📋 Step Details:")
    for step_name, step_success in steps:
        status = "✅" if step_success else "❌"
        print(f"  {status} {step_name}")

    print(f"\n📁 Results Location: {results_dir.absolute()}")
    print(f"📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if successful_steps == total_steps:
        print("\n🎉 All steps completed successfully!")
        return True
    else:
        print(f"\n⚠️  {total_steps - successful_steps} step(s) failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
