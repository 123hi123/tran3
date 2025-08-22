#!/usr/bin/env python3
"""
Complete ST-GCN pipeline runner
This script ensures all components work together with correct paths
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime


def write_fail_file(step_name, command, stdout_text, stderr_text, filename="fail.txt"):
    """Write a fail report file with useful debugging information."""
    try:
        lines = []
        lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Step: {step_name}")
        lines.append(f"Command: {command}")
        lines.append("")
        lines.append("=== STDOUT ===")
        lines.append(stdout_text or "<empty>")
        lines.append("")
        lines.append("=== STDERR ===")
        lines.append(stderr_text or "<empty>")
        lines.append("")
        content = "\n".join(lines)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"❌ Failure report written to {os.path.abspath(filename)}")
    except Exception as ex:
        print(f"⚠️ Failed to write fail.txt: {ex}")

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True, cwd=os.getcwd())
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        # Write a failure report for debugging
        write_fail_file(description, command, e.stdout, e.stderr)
        return False

def check_data_files():
    """Check if required data files exist"""
    train_csv = "data/raw/train.csv"
    val_csv = "data/raw/val.csv"
    
    if not os.path.exists(train_csv):
        print(f"❌ Missing: {train_csv}")
        return False
    
    if not os.path.exists(val_csv):
        print(f"❌ Missing: {val_csv}")
        return False
    
    print(f"✅ Found: {train_csv}")
    print(f"✅ Found: {val_csv}")
    return True

def main():
    """Run the complete ST-GCN pipeline"""
    parser = argparse.ArgumentParser(description='ST-GCN Pipeline Runner')
    parser.add_argument('--version', choices=['v1', 'v2'], default='v1',
                       help='Pipeline version to use (default: v1)')
    args = parser.parse_args()
    
    print(f"ST-GCN Pipeline Runner (Version: {args.version})")
    print("=" * 60)
    
    # Change to project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    print(f"Working directory: {os.getcwd()}")
    
    # Check Python path
    sys.path.insert(0, os.path.join(project_dir, 'src'))
    
    # Step 1: Check data files
    print("\n1. Checking data files...")
    if not check_data_files():
        print("\n❌ Please place your train.csv and val.csv files in data/raw/")
        print("   Refer to README.md for the expected format.")
        return False
    
    # Step 2: Data processing
    print("\n2. Processing data...")
    data_processor = "src/data_processor_v2.py" if args.version == 'v2' else "src/data_processor.py"
    if not run_command(f"python {data_processor}", "Data Processing"):
        print("❌ Data processing failed!")
        return False
    
    # Step 3: Model training
    print("\n3. Training model...")
    trainer = "src/train_v2.py" if args.version == 'v2' else "src/train.py"
    if not run_command(f"python {trainer}", "Model Training"):
        print("❌ Model training failed!")
        return False
    
    # Step 4: Model evaluation
    print("\n4. Evaluating model...")
    if not run_command("python src/evaluate.py", "Model Evaluation"):
        print("❌ Model evaluation failed!")
        return False
    
    print(f"\n{'='*60}")
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    print("\nResults location:")
    print(f"  - Trained models: models/")
    print(f"  - Evaluation results: results/")
    print(f"  - Processed data: data/processed/")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)