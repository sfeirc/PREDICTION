"""
Quickstart script to run the entire pipeline end-to-end.

This script will:
1. Fetch data from Binance
2. Engineer features
3. Train a transformer model
4. Evaluate on test set
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: str, description: str):
    """Run a command and handle errors."""
    print("\n" + "=" * 80)
    print(f"Step: {description}")
    print("=" * 80)
    print(f"Running: {cmd}\n")

    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        print(f"\nError: {description} failed with return code {result.returncode}")
        sys.exit(1)

    print(f"\n✓ {description} completed successfully")


def main():
    print("\n" + "=" * 80)
    print("Crypto Price Prediction - Quickstart Pipeline")
    print("=" * 80)

    # Check if config exists
    if not Path("config.yaml").exists():
        print("\nError: config.yaml not found")
        sys.exit(1)

    # Step 1: Fetch data for both symbols
    run_command(
        "python data_fetcher.py --symbol BTCUSDT --days 30",
        "Fetch BTCUSDT data",
    )

    run_command(
        "python data_fetcher.py --symbol ETHUSDT --days 30",
        "Fetch ETHUSDT data",
    )

    # Step 2: Feature engineering
    run_command(
        "python feature_engineering.py",
        "Feature engineering",
    )

    # Step 3: Train transformer model
    run_command(
        "python train.py --model transformer --no-wandb",
        "Train transformer model",
    )

    # Step 4: Evaluate
    run_command(
        "python evaluate.py --checkpoint checkpoints/best_model.pt",
        "Evaluate on test set",
    )

    # Optional: Regime analysis
    print("\n" + "=" * 80)
    print("Optional: Run regime analysis? (y/n)")
    print("=" * 80)
    response = input("> ").strip().lower()

    if response == "y":
        run_command(
            "python evaluate.py --checkpoint checkpoints/best_model.pt --regime-analysis",
            "Regime analysis",
        )

    print("\n" + "=" * 80)
    print("Pipeline Complete!")
    print("=" * 80)
    print("\nResults:")
    print("  - Logs: logs/")
    print("  - Checkpoints: checkpoints/")
    print("  - Plots: logs/*.png")
    print("\nNext steps:")
    print("  - Compare models: python train.py --model all")
    print("  - Ablation study: python evaluate.py --ablation")
    print("  - Test on ETH: python evaluate.py --test-symbol ETHUSDT")


if __name__ == "__main__":
    main()

