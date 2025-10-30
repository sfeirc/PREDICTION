"""
Enhanced training script with all improvements integrated.

Usage:
    python train_v2.py --model lightgbm
    python train_v2.py --model xgboost
    python train_v2.py --model random_forest
    python train_v2.py --model all
"""

import argparse
import sys
import io
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import yaml

# Fix encoding for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Local imports
from data_fetcher import BinanceDataFetcher
from feature_engineering_v2 import FeatureEngineV2
from datasets import TimeSeriesDataset, create_train_val_test_split
from models.baselines_v2 import BaselineModelV2
from utils import set_seed, setup_logging, get_device
from trading_simulator import TradingSimulator

# Optional W&B
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def train_baseline_v2(
    model_type: str,
    train_loader,
    val_loader,
    test_loader,
    feature_cols: list,
    config: Dict,
    use_wandb: bool = False,
):
    """Train enhanced baseline model."""
    print("\n" + "=" * 80)
    print(f"Training Enhanced Model: {model_type.upper()}")
    print("=" * 80)

    # Create model
    model = BaselineModelV2(model_type, config)

    # Train
    results = model.train(train_loader, val_loader)

    # Evaluate on test set
    print("\n" + "=" * 80)
    print("TEST SET EVALUATION")
    print("=" * 80)
    
    test_metrics = model.evaluate(
        test_loader,
        split="test",
        min_confidence=config["training"]["min_confidence"],
    )

    # Feature importance
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE")
    print("=" * 80)
    model.feature_importance(feature_cols, top_k=20)

    # Log to W&B
    if use_wandb:
        wandb.log({
            "test/accuracy": test_metrics["accuracy"],
            "test/auroc": test_metrics["auroc"],
            "test/f1": test_metrics["f1"],
            "test/precision": test_metrics["precision"],
            "test/recall": test_metrics["recall"],
        })
        
        if "high_confidence" in test_metrics:
            hc = test_metrics["high_confidence"]
            wandb.log({
                "test/high_conf_accuracy": hc["accuracy"],
                "test/high_conf_auroc": hc["auroc"],
            })

    return model, test_metrics


def run_trading_simulation(
    model,
    test_dataset,
    test_df,
    config: Dict,
):
    """Run trading simulation on test set."""
    print("\n" + "=" * 80)
    print("TRADING SIMULATION")
    print("=" * 80)

    # Get predictions
    X_test = test_dataset.feature_values
    y_test = test_dataset.targets
    y_pred, y_pred_proba = model.predict(X_test)

    # Get actual returns
    test_indices = test_df.index[config["sequence"]["lookback_minutes"]:]
    test_indices = test_indices[:len(y_pred)]
    actual_returns = test_df.loc[test_indices, "forward_return_5m"].values

    # Remove NaN
    valid_mask = ~np.isnan(actual_returns)
    y_pred_clean = y_pred[valid_mask]
    y_pred_proba_clean = y_pred_proba[valid_mask]
    actual_returns_clean = actual_returns[valid_mask]

    if len(y_pred_clean) == 0:
        print("No valid predictions for trading simulation")
        return None

    # Run simulation
    simulator = TradingSimulator(
        initial_capital=10000.0,
        trading_cost=config["evaluation"]["trading_cost"],
        min_confidence=config["training"]["min_confidence"],
    )

    sim_results = simulator.simulate(
        y_pred_clean,
        y_pred_proba_clean,
        actual_returns_clean,
    )

    simulator.print_results(sim_results)

    # Save plot
    plot_path = Path("logs") / "trading_simulation.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    simulator.plot_results(sim_results, save_path=str(plot_path))

    return sim_results


def main():
    parser = argparse.ArgumentParser(description="Train enhanced crypto price prediction model")
    parser.add_argument(
        "--model",
        type=str,
        default="lightgbm",
        choices=["logistic", "random_forest", "lightgbm", "xgboost", "all"],
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
    parser.add_argument(
        "--no-simulation",
        action="store_true",
        help="Skip trading simulation",
    )

    args = parser.parse_args()

    # Load config
    print("=" * 80)
    print("ENHANCED CRYPTO PRICE PREDICTION - TRAINING")
    print("=" * 80)
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Target thresholds: {config['target']['down_threshold']:.3f} to {config['target']['up_threshold']:.3f}")
    print(f"  Balance classes: {config['target']['balance_classes']}")
    print(f"  Min confidence: {config['training']['min_confidence']}")

    # Set seed
    set_seed(config["seed"])

    # Setup logging
    logger = setup_logging(
        log_dir=config["logging"]["log_dir"],
        experiment_name=f"{args.model}_v2",
    )

    # W&B
    use_wandb = config["logging"]["use_wandb"] and not args.no_wandb and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project=config["logging"]["wandb_project"],
            name=f"{args.model}_v2_{config['data']['train_symbol']}",
            config=config,
        )

    # Check if features already exist
    symbol = config["data"]["train_symbol"]
    processed_path = Path(config["data"]["processed_dir"]) / f"{symbol.lower()}_features_v2.parquet"

    if not processed_path.exists():
        print("\n" + "=" * 80)
        print("STEP 1: DATA FETCHING & FEATURE ENGINEERING")
        print("=" * 80)
        print("\nProcessed features not found. Running feature engineering...")

        # Fetch data
        fetcher = BinanceDataFetcher(cache_dir=config["data"]["cache_dir"])
        
        print(f"\nFetching {config['data']['train_symbol']}...")
        btc_df = fetcher.fetch_and_cache(
            symbol=config["data"]["train_symbol"],
            days=config["data"]["days"],
            interval=config["data"]["interval"],
        )

        # Fetch cross-asset if enabled
        eth_df = None
        if config["features"].get("use_cross_asset"):
            print(f"\nFetching {config['data']['test_symbol']} for cross-asset features...")
            eth_df = fetcher.fetch_and_cache(
                symbol=config["data"]["test_symbol"],
                days=config["data"]["days"],
                interval=config["data"]["interval"],
            )

        # Create features
        print("\nCreating enhanced features...")
        feature_engine = FeatureEngineV2(config)
        df = feature_engine.create_features(btc_df, cross_asset_df=eth_df)
        
        # Create targets
        df = feature_engine.create_multi_horizon_targets(
            df,
            horizons=config["target"]["horizons"],
            up_threshold=config["target"]["up_threshold"],
            down_threshold=config["target"]["down_threshold"],
        )

        # Balance classes
        if config["target"]["balance_classes"]:
            df = feature_engine.balance_classes(
                df,
                target_col="target_5m",
                method=config["target"]["balancing_method"],
            )

        # Save
        df.to_parquet(processed_path)
        print(f"\nSaved processed features to {processed_path}")
    else:
        print("\n" + "=" * 80)
        print("STEP 1: LOADING PROCESSED FEATURES")
        print("=" * 80)
        print(f"\nLoading features from {processed_path}")

    # Load processed features
    df = pd.read_parquet(processed_path)

    # Get feature columns
    exclude_cols = {
        "open", "high", "low", "close", "volume",
        "quote_volume", "trades", "taker_buy_base", "taker_buy_quote",
        "target_5m", "target_10m", "target_15m",
        "forward_return_5m", "forward_return_10m", "forward_return_15m",
        "hour", "minute", "day_of_week", "mid_price",
        "sample_weight", "regime", "rolling_vol",
    }
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"Total features: {len(feature_cols)}")

    # Split data
    print("\n" + "=" * 80)
    print("STEP 2: DATA SPLITTING")
    print("=" * 80)
    
    train_df, val_df, test_df = create_train_val_test_split(
        df,
        train_ratio=config["split"]["train"],
        val_ratio=config["split"]["val"],
        test_ratio=config["split"]["test"],
    )

    # Create datasets
    print("\n" + "=" * 80)
    print("STEP 3: DATASET CREATION")
    print("=" * 80)
    
    print("\nCreating flat datasets (for gradient boosting)...")
    
    from torch.utils.data import DataLoader
    
    train_dataset = TimeSeriesDataset(
        train_df,
        feature_cols,
        target_col="target_5m",
        lookback=config["sequence"]["lookback_minutes"],
        scaler=None,
    )

    val_dataset = TimeSeriesDataset(
        val_df,
        feature_cols,
        target_col="target_5m",
        lookback=config["sequence"]["lookback_minutes"],
        scaler=train_dataset.scaler,
    )

    test_dataset = TimeSeriesDataset(
        test_df,
        feature_cols,
        target_col="target_5m",
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

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")

    # Determine which models to train
    print("\n" + "=" * 80)
    print("STEP 4: MODEL TRAINING")
    print("=" * 80)
    
    if args.model == "all":
        models_to_train = ["logistic", "random_forest", "lightgbm", "xgboost"]
    else:
        models_to_train = [args.model]

    all_results = {}

    for model_type in models_to_train:
        try:
            # Train model
            model, test_metrics = train_baseline_v2(
                model_type,
                train_loader,
                val_loader,
                test_loader,
                feature_cols,
                config,
                use_wandb=use_wandb,
            )

            # Trading simulation
            if not args.no_simulation:
                sim_results = run_trading_simulation(
                    model,
                    test_dataset,
                    test_df,
                    config,
                )
                test_metrics["trading"] = sim_results

            all_results[model_type] = test_metrics

        except Exception as e:
            print(f"\nError training {model_type}: {e}")
            print("Skipping this model...")
            continue

    # Final summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE - RESULTS SUMMARY")
    print("=" * 80)

    if len(all_results) > 0:
        print("\nModel Comparison:")
        print(f"{'Model':<15} {'Accuracy':<10} {'AUROC':<10} {'F1':<10} {'Win Rate':<10} {'Return':<10}")
        print("-" * 80)

        for model_name, metrics in all_results.items():
            win_rate = metrics["trading"]["win_rate"] if "trading" in metrics else 0
            returns = metrics["trading"]["total_return_pct"] if "trading" in metrics else 0
            
            print(f"{model_name:<15} "
                  f"{metrics['accuracy']:<10.4f} "
                  f"{metrics['auroc']:<10.4f} "
                  f"{metrics['f1']:<10.4f} "
                  f"{win_rate:<10.2%} "
                  f"{returns:<10.2f}%")

        # Best model
        best_model = max(all_results.items(), key=lambda x: x[1]["auroc"])
        print(f"\n🏆 Best Model: {best_model[0].upper()} (AUROC: {best_model[1]['auroc']:.4f})")

    if use_wandb:
        wandb.finish()

    print("\n" + "=" * 80)
    print("All done! Check logs/ directory for detailed results.")
    print("=" * 80)


if __name__ == "__main__":
    main()

