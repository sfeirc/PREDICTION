"""
Simple profitable training - matches find_optimal_confidence.py exactly
"""

import sys, io
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from datasets import TimeSeriesDataset, create_train_val_test_split
from models.baselines_v2 import BaselineModelV2
from trading_simulator import TradingSimulator
from torch.utils.data import DataLoader
from utils import set_seed

print("=" * 80)
print("💰 SIMPLE PROFITABLE TRAINING")
print("=" * 80)

# Load config
with open("config_profitable.yaml") as f:
    config = yaml.safe_load(f)

set_seed(config["seed"])

# Load data
processed_path = Path(config["data"]["processed_dir"]) / "btcusdt_features_v2.parquet"
df = pd.read_parquet(processed_path)

# Get features
exclude_cols = {"open", "high", "low", "close", "volume", "quote_volume", "trades", 
               "taker_buy_base", "taker_buy_quote", "target_5m", "target_10m", "target_15m",
               "forward_return_5m", "forward_return_10m", "forward_return_15m",
               "hour", "minute", "day_of_week", "mid_price", "sample_weight", "regime", "rolling_vol"}
feature_cols = [col for col in df.columns if col not in exclude_cols]

print(f"\nFeatures: {len(feature_cols)}")
print(f"Total samples: {len(df)}")

# Split
train_df, val_df, test_df = create_train_val_test_split(df, 0.70, 0.15, 0.15)

# Create datasets (NO FILTERING - like find_optimal)
train_dataset = TimeSeriesDataset(train_df, feature_cols, "target_5m", config["sequence"]["lookback_minutes"])
val_dataset = TimeSeriesDataset(val_df, feature_cols, "target_5m", config["sequence"]["lookback_minutes"], train_dataset.scaler)
test_dataset = TimeSeriesDataset(test_df, feature_cols, "target_5m", config["sequence"]["lookback_minutes"], train_dataset.scaler)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

# Train
print("\nTraining LightGBM...")
model = BaselineModelV2("lightgbm", config)
model.train(train_loader, val_loader)

# Get predictions
X_test = test_dataset.feature_values
y_pred, y_pred_proba = model.predict(X_test)

test_indices = test_df.index[config["sequence"]["lookback_minutes"]:][:len(y_pred)]
actual_returns = test_df.loc[test_indices, "forward_return_5m"].values

valid_mask = ~np.isnan(actual_returns)
y_pred = y_pred[valid_mask]
y_pred_proba = y_pred_proba[valid_mask]
actual_returns = actual_returns[valid_mask]

print(f"Valid predictions: {len(y_pred)}")

# Debug: Check what we're passing to simulator
print(f"\nDEBUG INFO:")
print(f"  y_pred shape: {y_pred.shape}")
print(f"  y_pred_proba shape: {y_pred_proba.shape}")
print(f"  actual_returns shape: {actual_returns.shape}")
print(f"  actual_returns has NaN: {np.isnan(actual_returns).sum()}")
print(f"  Predictions sum (1=up): {y_pred.sum()}")

# Check confidence distribution
confidences = np.maximum(y_pred_proba, 1 - y_pred_proba)
print(f"  Predictions with conf >= 0.70: {(confidences >= 0.70).sum()}")
print(f"  Predictions with conf >= 0.75: {(confidences >= 0.75).sum()}")

# Test at 70% and 75%
print("\n" + "=" * 80)
print("TESTING PROFITABILITY")
print("=" * 80)

for conf in [0.70, 0.75]:
    simulator = TradingSimulator(
        initial_capital=10000,
        trading_cost=config["evaluation"]["trading_cost"],
        min_confidence=conf,
    )
    
    results = simulator.simulate(y_pred, y_pred_proba, actual_returns)
    
    print(f"\nConfidence: {conf:.0%}")
    print(f"  Trades:    {results['n_trades']}")
    print(f"  Win Rate:  {results['win_rate']:.1%}")
    print(f"  Return:    {results['total_return_pct']:+.2f}%")
    print(f"  Final:     ${results['final_capital']:,.2f}")
    
    if results['total_return_pct'] > 0:
        print(f"  ✅ PROFITABLE!")

print("\n" + "=" * 80)
print("Complete! Use confidence 0.75 for best results.")
print("=" * 80)

