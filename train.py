"""
Main training script for crypto price prediction models.

Supports:
- Baseline models (Logistic Regression, Random Forest)
- Sequence models (LSTM, Transformer)
- Time-series train/val/test split
- Early stopping
- Learning rate scheduling
- Weights & Biases logging
- Model checkpointing
"""

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

# Local imports
from datasets import TimeSeriesDataset, SequenceDataset, create_train_val_test_split
from models.baselines import BaselineModel
from models.seq import create_model, count_parameters
from utils import (
    set_seed,
    load_config,
    setup_logging,
    save_checkpoint,
    get_device,
    EarlyStopping,
    print_model_summary,
    plot_training_history,
)

# Optional W&B
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not available, skipping experiment tracking")


class Trainer:
    """Trainer for sequence models (LSTM, Transformer)."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: torch.device,
        logger=None,
        use_wandb: bool = False,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.logger = logger
        self.use_wandb = use_wandb

        # Loss function (class weighted for imbalanced data)
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )

        # Learning rate scheduler
        if config["training"]["lr_scheduler"]:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=config["training"]["scheduler_factor"],
                patience=config["training"]["scheduler_patience"],
                verbose=True,
            )
        else:
            self.scheduler = None

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config["training"]["early_stopping_patience"],
            mode="max",
        )

        # Training history
        self.history = {
            "train_loss": [],
            "train_accuracy": [],
            "train_auroc": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_auroc": [],
        }

        self.best_val_auroc = 0.0

    def train_epoch(self) -> Dict:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = []

        pbar = tqdm(self.train_loader, desc="Training")

        for features, targets in pbar:
            features = features.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            outputs = self.model(features)
            loss = self.criterion(outputs, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.config["training"]["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config["training"]["grad_clip"],
                )

            self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())

            pbar.set_postfix({"loss": loss.item()})

        # Compute metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = (np.array(all_preds) == np.array(all_targets)).mean()

        # AUROC
        from sklearn.metrics import roc_auc_score

        auroc = roc_auc_score(all_targets, all_probs)

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "auroc": auroc,
        }

    def validate(self) -> Dict:
        """Validate on validation set."""
        self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for features, targets in tqdm(self.val_loader, desc="Validation"):
                features = features.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                outputs = self.model(features)
                loss = self.criterion(outputs, targets)

                # Metrics
                total_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)[:, 1]
                preds = outputs.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # Compute metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = (np.array(all_preds) == np.array(all_targets)).mean()

        from sklearn.metrics import roc_auc_score

        auroc = roc_auc_score(all_targets, all_probs)

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "auroc": auroc,
        }

    def train(self, epochs: int):
        """Full training loop."""
        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60)

        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            print("-" * 60)

            # Train
            train_metrics = self.train_epoch()
            print(
                f"Train - Loss: {train_metrics['loss']:.4f}, "
                f"Accuracy: {train_metrics['accuracy']:.4f}, "
                f"AUROC: {train_metrics['auroc']:.4f}"
            )

            # Validate
            val_metrics = self.validate()
            print(
                f"Val   - Loss: {val_metrics['loss']:.4f}, "
                f"Accuracy: {val_metrics['accuracy']:.4f}, "
                f"AUROC: {val_metrics['auroc']:.4f}"
            )

            # Update history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_accuracy"].append(train_metrics["accuracy"])
            self.history["train_auroc"].append(train_metrics["auroc"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_accuracy"].append(val_metrics["accuracy"])
            self.history["val_auroc"].append(val_metrics["auroc"])

            # Log to W&B
            if self.use_wandb:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train/loss": train_metrics["loss"],
                        "train/accuracy": train_metrics["accuracy"],
                        "train/auroc": train_metrics["auroc"],
                        "val/loss": val_metrics["loss"],
                        "val/accuracy": val_metrics["accuracy"],
                        "val/auroc": val_metrics["auroc"],
                        "lr": self.optimizer.param_groups[0]["lr"],
                    }
                )

            # Learning rate scheduler
            if self.scheduler is not None:
                self.scheduler.step(val_metrics["auroc"])

            # Save best model
            if val_metrics["auroc"] > self.best_val_auroc:
                self.best_val_auroc = val_metrics["auroc"]
                print(f"New best AUROC: {self.best_val_auroc:.4f}")

                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_metrics,
                    self.config,
                    checkpoint_dir=self.config["logging"]["checkpoint_dir"],
                    filename="best_model.pt",
                )

            # Early stopping
            if self.early_stopping(val_metrics["auroc"]):
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break

        print("\n" + "=" * 60)
        print("Training Complete")
        print("=" * 60)
        print(f"Best validation AUROC: {self.best_val_auroc:.4f}")

        return self.history


def train_baseline(
    model_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    feature_cols: list,
    config: Dict,
    use_wandb: bool = False,
):
    """Train baseline model (Logistic Regression or Random Forest)."""
    print("\n" + "=" * 60)
    print(f"Training Baseline Model: {model_type}")
    print("=" * 60)

    # Create model
    model = BaselineModel(model_type, config)

    # Train
    results = model.train(train_loader, val_loader)

    # Evaluate on test set
    test_metrics = model.evaluate(test_loader, split="test")

    # Feature importance (for Random Forest)
    if model_type == "random_forest":
        model.feature_importance(feature_cols, top_k=20)

    # Log to W&B
    if use_wandb:
        wandb.log(
            {
                "test/accuracy": test_metrics["accuracy"],
                "test/auroc": test_metrics["auroc"],
                "test/f1": test_metrics["f1"],
                "test/precision": test_metrics["precision"],
                "test/recall": test_metrics["recall"],
            }
        )

    return model, test_metrics


def train_sequence_model(
    model_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    input_size: int,
    config: Dict,
    use_wandb: bool = False,
):
    """Train sequence model (LSTM or Transformer)."""
    print("\n" + "=" * 60)
    print(f"Training Sequence Model: {model_type}")
    print("=" * 60)

    # Device
    device = get_device()

    # Create model
    model = create_model(model_type, input_size, config)
    print_model_summary(model)

    # Trainer
    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        config,
        device,
        use_wandb=use_wandb,
    )

    # Train
    history = trainer.train(epochs=config["training"]["epochs"])

    # Plot training history
    plot_path = Path("logs") / f"{model_type}_training_history.png"
    plot_training_history(history, save_path=str(plot_path))

    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train crypto price prediction model")
    parser.add_argument(
        "--model",
        type=str,
        default="transformer",
        choices=["logistic", "random_forest", "lstm", "transformer", "all"],
        help="Model type to train",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Set seed
    set_seed(config["seed"])

    # Setup logging
    logger = setup_logging(
        log_dir=config["logging"]["log_dir"],
        experiment_name=f"{args.model}_{Path().cwd().name}",
    )

    # W&B
    use_wandb = config["logging"]["use_wandb"] and not args.no_wandb and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project=config["logging"]["wandb_project"],
            name=f"{args.model}_{config['data']['train_symbol']}",
            config=config,
        )

    # Load processed features
    symbol = config["data"]["train_symbol"]
    processed_path = Path(config["data"]["processed_dir"]) / f"{symbol.lower()}_features.parquet"

    if not processed_path.exists():
        print(f"\nProcessed features not found at {processed_path}")
        print("Running feature engineering first...")

        from data_fetcher import BinanceDataFetcher
        from feature_engineering import FeatureEngine

        # Fetch data
        fetcher = BinanceDataFetcher(cache_dir=config["data"]["cache_dir"])
        df = fetcher.fetch_and_cache(
            symbol=symbol,
            days=config["data"]["days"],
            interval=config["data"]["interval"],
        )

        # Create features
        feature_engine = FeatureEngine(config)
        df = feature_engine.create_features(df)
        df = feature_engine.create_target(
            df,
            forward_minutes=config["target"]["forward_minutes"],
            up_threshold=config["target"]["up_threshold"],
            down_threshold=config["target"]["down_threshold"],
        )

        # Save
        df.to_parquet(processed_path)
        print(f"Saved processed features to {processed_path}")

    print(f"\nLoading features from {processed_path}")
    import pandas as pd

    df = pd.read_parquet(processed_path)

    # Get feature columns
    exclude_cols = {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
        "target",
        "forward_return",
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

    # Determine which models to train
    models_to_train = [args.model] if args.model != "all" else ["logistic", "random_forest", "lstm", "transformer"]

    for model_type in models_to_train:
        if model_type in ["logistic", "random_forest"]:
            # Baseline models - use flat datasets
            print("\nCreating flat datasets...")

            train_dataset = TimeSeriesDataset(
                train_df,
                feature_cols,
                target_col="target",
                lookback=config["sequence"]["lookback_minutes"],
                scaler=None,
            )

            val_dataset = TimeSeriesDataset(
                val_df,
                feature_cols,
                target_col="target",
                lookback=config["sequence"]["lookback_minutes"],
                scaler=train_dataset.scaler,
            )

            test_dataset = TimeSeriesDataset(
                test_df,
                feature_cols,
                target_col="target",
                lookback=config["sequence"]["lookback_minutes"],
                scaler=train_dataset.scaler,
            )

            # Data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=config["training"]["batch_size"],
                shuffle=False,  # Keep time order
            )
            val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"], shuffle=False)

            # Train
            model, test_metrics = train_baseline(
                model_type,
                train_loader,
                val_loader,
                test_loader,
                feature_cols,
                config,
                use_wandb=use_wandb,
            )

        else:
            # Sequence models - use sequence datasets
            print("\nCreating sequence datasets...")

            train_dataset = SequenceDataset(
                train_df,
                feature_cols,
                target_col="target",
                lookback=config["sequence"]["lookback_minutes"],
                scaler=None,
            )

            val_dataset = SequenceDataset(
                val_df,
                feature_cols,
                target_col="target",
                lookback=config["sequence"]["lookback_minutes"],
                scaler=train_dataset.scaler,
            )

            test_dataset = SequenceDataset(
                test_df,
                feature_cols,
                target_col="target",
                lookback=config["sequence"]["lookback_minutes"],
                scaler=train_dataset.scaler,
            )

            # Data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=config["training"]["batch_size"],
                shuffle=False,
            )
            val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"], shuffle=False)

            # Train
            model, history = train_sequence_model(
                model_type,
                train_loader,
                val_loader,
                test_loader,
                input_size=len(feature_cols),
                config=config,
                use_wandb=use_wandb,
            )

    if use_wandb:
        wandb.finish()

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

