"""
Evaluation script with regime analysis and ablation studies.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

from datasets import SequenceDataset, TimeSeriesDataset, create_train_val_test_split
from models.seq import create_model
from utils import load_config, load_checkpoint, get_device, plot_confusion_matrix, plot_roc_curve


class ModelEvaluator:
    """Evaluate trained models with various analyses."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: Dict,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.config = config

    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions.

        Returns:
            (predictions, probabilities, targets)
        """
        all_preds = []
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for features, targets in data_loader:
                features = features.to(self.device)

                outputs = self.model(features)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                preds = outputs.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_targets.extend(targets.numpy())

        return (
            np.array(all_preds),
            np.array(all_probs),
            np.array(all_targets),
        )

    def evaluate(self, data_loader: DataLoader, split_name: str = "test") -> Dict:
        """
        Evaluate model and compute metrics.

        Args:
            data_loader: Data loader
            split_name: Name of split (for logging)

        Returns:
            Dictionary with metrics
        """
        print(f"\n{'=' * 60}")
        print(f"Evaluating on {split_name} set")
        print("=" * 60)

        preds, probs, targets = self.predict(data_loader)

        # Compute metrics
        metrics = {
            "accuracy": accuracy_score(targets, preds),
            "auroc": roc_auc_score(targets, probs),
            "f1": f1_score(targets, preds),
            "precision": precision_score(targets, preds, zero_division=0),
            "recall": recall_score(targets, preds, zero_division=0),
        }

        # Print metrics
        print(f"\n{split_name.capitalize()} Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric:15s}: {value:.4f}")

        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(targets, preds, target_names=["Down", "Up"]))

        # Baseline comparison (random 50% classifier)
        random_accuracy = 0.5
        random_auroc = 0.5
        print(f"\nComparison to Random Baseline:")
        print(f"  Accuracy improvement: {(metrics['accuracy'] - random_accuracy):.4f} ({(metrics['accuracy'] / random_accuracy - 1) * 100:+.2f}%)")
        print(f"  AUROC improvement:    {(metrics['auroc'] - random_auroc):.4f} ({(metrics['auroc'] / random_auroc - 1) * 100:+.2f}%)")

        # Save plots
        plot_dir = Path("logs")
        plot_dir.mkdir(exist_ok=True)

        plot_confusion_matrix(
            targets,
            preds,
            save_path=str(plot_dir / f"confusion_matrix_{split_name}.png"),
        )

        plot_roc_curve(
            targets,
            probs,
            save_path=str(plot_dir / f"roc_curve_{split_name}.png"),
        )

        return metrics

    def regime_analysis(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        dataset_class,
        scaler,
    ) -> Dict:
        """
        Perform regime-based analysis (high vs low volatility).

        Args:
            df: DataFrame with features and regime labels
            feature_cols: List of feature column names
            dataset_class: Dataset class to use
            scaler: Pre-fitted scaler

        Returns:
            Dictionary with regime-specific metrics
        """
        print("\n" + "=" * 60)
        print("Regime-Based Analysis")
        print("=" * 60)

        if "regime" not in df.columns:
            print("Computing volatility regimes...")
            df = self._compute_regimes(df)

        results = {}

        for regime in ["low", "mid", "high"]:
            regime_df = df[df["regime"] == regime].copy()

            if len(regime_df) < 100:
                print(f"\nSkipping {regime} volatility regime (too few samples: {len(regime_df)})")
                continue

            print(f"\n{regime.upper()} Volatility Regime ({len(regime_df)} samples)")
            print("-" * 60)

            # Create dataset
            dataset = dataset_class(
                regime_df,
                feature_cols,
                target_col="target",
                lookback=self.config["sequence"]["lookback_minutes"],
                scaler=scaler,
            )

            if len(dataset) == 0:
                print(f"No valid samples in {regime} volatility regime")
                continue

            loader = DataLoader(dataset, batch_size=256, shuffle=False)

            # Evaluate
            preds, probs, targets = self.predict(loader)

            metrics = {
                "accuracy": accuracy_score(targets, preds),
                "auroc": roc_auc_score(targets, probs),
                "f1": f1_score(targets, preds),
                "precision": precision_score(targets, preds, zero_division=0),
                "recall": recall_score(targets, preds, zero_division=0),
            }

            for metric, value in metrics.items():
                print(f"  {metric:15s}: {value:.4f}")

            results[regime] = metrics

        # Compare regimes
        print("\n" + "=" * 60)
        print("Regime Comparison")
        print("=" * 60)

        for metric in ["accuracy", "auroc", "f1"]:
            print(f"\n{metric.upper()}:")
            for regime in ["low", "mid", "high"]:
                if regime in results:
                    print(f"  {regime:5s}: {results[regime][metric]:.4f}")

        return results

    def _compute_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute volatility regimes (low, mid, high)."""
        # Compute log returns
        log_returns = np.log(df["close"] / df["close"].shift(1))

        # Rolling volatility (24 hours = 1440 minutes)
        window = self.config["regime"]["volatility_window"]
        rolling_vol = log_returns.rolling(window=window).std()

        # Percentiles
        low_pct = self.config["regime"]["low_percentile"]
        high_pct = self.config["regime"]["high_percentile"]

        low_threshold = rolling_vol.quantile(low_pct / 100)
        high_threshold = rolling_vol.quantile(high_pct / 100)

        # Assign regimes
        df["rolling_vol"] = rolling_vol
        df["regime"] = "mid"
        df.loc[rolling_vol <= low_threshold, "regime"] = "low"
        df.loc[rolling_vol >= high_threshold, "regime"] = "high"

        # Count regimes
        regime_counts = df["regime"].value_counts()
        print(f"\nRegime distribution:")
        for regime, count in regime_counts.items():
            print(f"  {regime:5s}: {count:6d} ({count / len(df) * 100:5.2f}%)")

        return df


def ablation_study(
    config: Dict,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_groups: Dict[str, List[str]],
):
    """
    Perform ablation study on feature groups.

    Args:
        config: Configuration dictionary
        train_df: Training DataFrame
        test_df: Test DataFrame
        feature_groups: Dictionary mapping group names to feature patterns
    """
    print("\n" + "=" * 60)
    print("Feature Ablation Study")
    print("=" * 60)

    device = get_device()
    results = {}

    for group_name, patterns in feature_groups.items():
        print(f"\n{'-' * 60}")
        print(f"Feature Group: {group_name}")
        print(f"Patterns: {patterns}")
        print("-" * 60)

        # Get feature columns matching patterns
        all_cols = [col for col in train_df.columns if col not in {
            "open", "high", "low", "close", "volume",
            "quote_volume", "trades", "taker_buy_base", "taker_buy_quote",
            "target", "forward_return", "regime", "rolling_vol",
        }]

        if patterns == ["all"]:
            feature_cols = all_cols
        else:
            feature_cols = [
                col for col in all_cols
                if any(pattern in col for pattern in patterns)
            ]

        print(f"Using {len(feature_cols)} features")

        if len(feature_cols) == 0:
            print("No features found, skipping...")
            continue

        # Create datasets
        train_dataset = SequenceDataset(
            train_df,
            feature_cols,
            target_col="target",
            lookback=config["sequence"]["lookback_minutes"],
            scaler=None,
        )

        test_dataset = SequenceDataset(
            test_df,
            feature_cols,
            target_col="target",
            lookback=config["sequence"]["lookback_minutes"],
            scaler=train_dataset.scaler,
        )

        if len(train_dataset) == 0 or len(test_dataset) == 0:
            print("Empty dataset, skipping...")
            continue

        # Create loaders
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

        # Train simple model (just for ablation comparison)
        print(f"Training simple LSTM for ablation...")

        model = create_model("lstm", len(feature_cols), config)
        model = model.to(device)

        # Quick training (just a few epochs for comparison)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(5):  # Just 5 epochs for quick comparison
            model.train()
            for features, targets in train_loader:
                features = features.to(device)
                targets = targets.to(device)

                outputs = model(features)
                loss = criterion(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Evaluate
        model.eval()
        all_preds = []
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for features, targets in test_loader:
                features = features.to(device)

                outputs = model(features)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                preds = outputs.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_targets.extend(targets.numpy())

        preds = np.array(all_preds)
        probs = np.array(all_probs)
        targets = np.array(all_targets)

        # Metrics
        metrics = {
            "accuracy": accuracy_score(targets, preds),
            "auroc": roc_auc_score(targets, probs),
            "f1": f1_score(targets, preds),
        }

        results[group_name] = metrics

        print(f"Results:")
        for metric, value in metrics.items():
            print(f"  {metric:15s}: {value:.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("Ablation Study Summary")
    print("=" * 60)

    for metric in ["accuracy", "auroc", "f1"]:
        print(f"\n{metric.upper()}:")
        for group_name, metrics in results.items():
            print(f"  {group_name:20s}: {metrics[metric]:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate crypto price prediction model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--test-symbol",
        type=str,
        default=None,
        help="Symbol to test on (if different from training symbol)",
    )
    parser.add_argument(
        "--regime-analysis",
        action="store_true",
        help="Perform regime-based analysis",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Perform feature ablation study",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Device
    device = get_device()

    # Test symbol (use train symbol if not specified)
    test_symbol = args.test_symbol or config["data"]["test_symbol"]

    print(f"\nTest symbol: {test_symbol}")

    # Load processed features
    processed_path = Path(config["data"]["processed_dir"]) / f"{test_symbol.lower()}_features.parquet"

    if not processed_path.exists():
        print(f"\nProcessed features not found for {test_symbol}")
        print("Run feature engineering first")
        return

    print(f"Loading features from {processed_path}")
    df = pd.read_parquet(processed_path)

    # Get feature columns
    exclude_cols = {
        "open", "high", "low", "close", "volume",
        "quote_volume", "trades", "taker_buy_base", "taker_buy_quote",
        "target", "forward_return",
    }
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    print(f"Number of features: {len(feature_cols)}")

    # Split data
    train_df, val_df, test_df = create_train_val_test_split(
        df,
        train_ratio=config["split"]["train"],
        val_ratio=config["split"]["val"],
        test_ratio=config["split"]["test"],
    )

    if args.ablation:
        # Ablation study
        feature_groups = {
            "returns_only": ["return_"],
            "volatility_only": ["volatility", "parkinson"],
            "volume_only": ["volume_", "taker_buy"],
            "indicators_only": ["rsi", "bb_", "vwap"],
            "returns+volatility": ["return_", "volatility", "parkinson"],
            "all_features": ["all"],
        }

        ablation_study(config, train_df, test_df, feature_groups)

    else:
        # Load checkpoint
        checkpoint_path = Path(args.checkpoint)

        if not checkpoint_path.exists():
            print(f"Checkpoint not found: {checkpoint_path}")
            return

        # Determine model type from config
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_config = checkpoint.get("config", config)

        # Create model (assume transformer for now, can be made smarter)
        print("\nCreating model...")
        model = create_model("transformer", len(feature_cols), model_config)

        # Load checkpoint
        load_checkpoint(str(checkpoint_path), model)

        # Create evaluator
        evaluator = ModelEvaluator(model, device, config)

        # Create test dataset
        test_dataset = SequenceDataset(
            test_df,
            feature_cols,
            target_col="target",
            lookback=config["sequence"]["lookback_minutes"],
            scaler=None,
        )

        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

        # Evaluate
        test_metrics = evaluator.evaluate(test_loader, split_name="test")

        # Regime analysis
        if args.regime_analysis:
            regime_results = evaluator.regime_analysis(
                test_df,
                feature_cols,
                SequenceDataset,
                test_dataset.scaler,
            )


if __name__ == "__main__":
    main()

