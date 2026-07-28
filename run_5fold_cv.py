#!/usr/bin/env python
"""5-fold cross-validation for ORGANT2 dataset.
Runs each fold as a separate subprocess to avoid Sacred global state issues.
"""
import subprocess
import sys
import os

# Configuration
N_FOLDS = 5
CONFIG_FILE = "configs/organt2_vit.yml"
TOTAL_CASES = 20
CASES_PER_FOLD = TOTAL_CASES // N_FOLDS  # 4 cases per fold

def run_fold(fold_id):
    """Run a single fold: train on 16 cases, test on 4 cases."""
    # Calculate train/test split for this fold
    test_start = fold_id * CASES_PER_FOLD
    test_end = test_start + CASES_PER_FOLD
    
    print(f"\n{'='*60}")
    print(f"Fold {fold_id}: test cases [{test_start}, {test_end})")
    print(f"{'='*60}\n")
    
    # Command to run this fold
    cmd = [
        sys.executable, "run.py", "train",
        "with", CONFIG_FILE,
        f"split={fold_id}",
        f"fold_id={fold_id}",
        f"test_fold_start={test_start}",
        f"test_fold_end={test_end}",
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    
    if result.returncode != 0:
        print(f"Fold {fold_id} failed with return code {result.returncode}")
        return None
    
    return fold_id

def main():
    print("Starting 5-fold cross-validation for ORGANT2")
    print(f"Total cases: {TOTAL_CASES}, Cases per fold: {CASES_PER_FOLD}")
    
    results = []
    for fold in range(N_FOLDS):
        result = run_fold(fold)
        if result is not None:
            results.append(result)
    
    print(f"\n{'='*60}")
    print(f"Completed {len(results)}/{N_FOLDS} folds successfully")
    print(f"Successful folds: {results}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
