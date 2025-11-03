"""
Cleanup script to remove unused/old files
Run with: python cleanup_unused.py
"""
import os
import shutil
from pathlib import Path

# Files to remove
files_to_remove = [
    # Old training scripts
    "train.py",
    "train_v2.py",
    "train_profitable.py",
    "train_simple_profitable.py",
    
    # Old feature engineering
    "feature_engineering.py",
    "feature_engineering_v2.py",
    "integrate_ultimate_features.py",
    
    # Old data fetching
    "data_fetcher.py",
    
    # Old models
    "models/baselines.py",
    "models/baselines_v2.py",
    
    # Old evaluation
    "evaluate.py",
    
    # Test/debug scripts
    "test_1hour.py",
    "test_pipeline.py",
    "test_worldclass_simple.py",
    "test_and_report.py",
    "test_confidence_boost.py",
    "quick_debug.py",
    "diagnose_confidence.py",
    "analyze_predictions.py",
    
    # Optimization scripts
    "find_optimal_confidence.py",
    "find_optimal_tradeoff.py",
    "maximize_profits.py",
    "maximize_trades_ultimate.py",
    "fetch_max_data.py",
    "fetch_max_data_unbalanced.py",
    
    # Demo scripts
    "DEMO_worldclass.py",
    
    # Old configs
    "config.yaml",
    "config_profitable.yaml",
    "config_worldclass.yaml",
    
    # Old batch files
    "RUN_1HOUR_TEST.bat",
    "RUN_COMPLETE_TEST.bat",
    "RUN_ULTIMATE_TEST.bat",
    
    # Documentation (consolidated into README.md)
    "README_WORLDCLASS.md",
    "QUICKSTART.md",
    "QUICK_START_TRADING.md",
    "START_HERE_ULTIMATE.md",
    "ULTIMATE_COMPLETE.md",
    "MISSION_ACCOMPLISHED.md",
    "100_PERCENT_COMPLETE.md",
    "BUILD_STATUS_REPORT.md",
    "ULTIMATE_ENHANCEMENTS.md",
    "QUICK_PROFIT_BOOSTERS.md",
    "NUCLEAR_BUILD_PLAN.md",
    "IMPROVEMENTS.md",
    "FINAL_SUMMARY.txt",
    "WORLDCLASS_COMPLETE.txt",
    "IMPLEMENTATION_COMPLETE.txt",
    "IMPLEMENTATION_SUMMARY.txt",
    "STATUS_READY.md",
    "READY_FOR_GITHUB.md",
    "GITHUB_SETUP.md",
    "NEXT_STEPS.md",
    "TRADING_STATUS_REPORT.md",
    "FIXED_MEMORY_AND_PREDICTIONS.md",
    "FIXED_PREDICTIONS_INFO.md",
    "ADAPTIVE_OPTIMIZATION_EXPLAINED.md",
    "_READ_ME_FIRST.txt",
    
    # Utility scripts (optional)
    "quickstart.py",
    "verify_setup.py",
    "run_wf_fast.py",
    "worldclass_bot.py",
]

# Directories to remove
dirs_to_remove = [
    "catboost_info",
    "wandb",
]

def cleanup():
    removed_files = []
    removed_dirs = []
    errors = []
    
    print("Cleaning up unused files...\n")
    
    # Remove files
    for file_path in files_to_remove:
        path = Path(file_path)
        if path.exists():
            try:
                if path.is_file():
                    path.unlink()
                    removed_files.append(file_path)
                    print(f"Removed: {file_path}")
                elif path.is_dir():
                    shutil.rmtree(path)
                    removed_dirs.append(file_path)
                    print(f"Removed directory: {file_path}")
            except Exception as e:
                errors.append((file_path, str(e)))
                print(f"Error removing {file_path}: {e}")
        else:
            print(f"Skipped (not found): {file_path}")
    
    # Remove directories
    for dir_path in dirs_to_remove:
        path = Path(dir_path)
        if path.exists() and path.is_dir():
            try:
                shutil.rmtree(path)
                removed_dirs.append(dir_path)
                print(f"Removed directory: {dir_path}")
            except Exception as e:
                errors.append((dir_path, str(e)))
                print(f"Error removing {dir_path}: {e}")
    
    print("\n" + "="*60)
    print(f"Summary:")
    print(f"  Files removed: {len(removed_files)}")
    print(f"  Directories removed: {len(removed_dirs)}")
    print(f"  Errors: {len(errors)}")
    
    if errors:
        print("\nErrors encountered:")
        for item, error in errors:
            print(f"  {item}: {error}")
    
    print("\nCleanup complete!")
    print("\nNote: Review removed files list before running if unsure.")
    print("You can restore from git if needed: git checkout <file>")

if __name__ == "__main__":
    response = input("This will remove many files. Continue? (yes/no): ")
    if response.lower() == "yes":
        cleanup()
    else:
        print("Cancelled.")

