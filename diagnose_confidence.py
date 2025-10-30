"""
Diagnose why model isn't confident
"""

import sys, io
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from datasets import TimeSeriesDataset, create_train_val_test_split
from models.baselines_v2 import BaselineModelV2
from torch.utils.data import DataLoader
from utils import set_seed

# Load config
with open("config_profitable.yaml") as f:
    config = yaml.safe_load(f)

set_seed(config["seed"])

# Load data
processed_path = Path(config["data"]["processed_dir"]) / "btcusdt_features_v2.parquet"
df = pd.read_parquet(processed_path)

exclude_cols = {"open", "high", "low", "close", "volume", "quote_volume", "trades", 
               "taker_buy_base", "taker_buy_quote", "target_5m", "target_10m", "target_15m",
               "forward_return_5m", "forward_return_10m", "forward_return_15m",
               "hour", "minute", "day_of_week", "mid_price", "sample_weight", "regime", "rolling_vol"}
feature_cols = [col for col in df.columns if col not in exclude_cols]

# Split
train_df, val_df, test_df = create_train_val_test_split(df, 0.70, 0.15, 0.15)

# Create datasets
train_dataset = TimeSeriesDataset(train_df, feature_cols, "target_5m", config["sequence"]["lookback_minutes"])
test_dataset = TimeSeriesDataset(test_df, feature_cols, "target_5m", config["sequence"]["lookback_minutes"], train_dataset.scaler)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

# Train
print("Training LightGBM...")
model = BaselineModelV2("lightgbm", config)
model.train(train_loader, None)

# Get predictions
X_test = test_dataset.feature_values
y_pred, y_pred_proba = model.predict(X_test)

print("\n" + "=" * 80)
print("CONFIDENCE DISTRIBUTION")
print("=" * 80)

print(f"\nPrediction probabilities (for class 1):")
print(f"  Min:    {y_pred_proba.min():.4f}")
print(f"  Max:    {y_pred_proba.max():.4f}")
print(f"  Mean:   {y_pred_proba.mean():.4f}")
print(f"  Median: {np.median(y_pred_proba):.4f}")

print(f"\nConfidence distribution (max probability):")
confidences = np.maximum(y_pred_proba, 1 - y_pred_proba)
print(f"  Min:    {confidences.min():.4f}")
print(f"  Max:    {confidences.max():.4f}")
print(f"  Mean:   {confidences.mean():.4f}")
print(f"  Median: {np.median(confidences):.4f}")

print(f"\nHow many predictions pass confidence thresholds:")
for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
    n_pass = (confidences >= threshold).sum()
    pct = n_pass / len(confidences) * 100
    print(f"  ≥ {threshold:.0%}: {n_pass:4d} / {len(confidences)} ({pct:5.1f}%)")

print(f"\nTop 10 most confident predictions:")
top_indices = np.argsort(confidences)[::-1][:10]
for i, idx in enumerate(top_indices, 1):
    print(f"  {i:2d}. Confidence: {confidences[idx]:.4f}, Pred: {y_pred[idx]}, Prob: {y_pred_proba[idx]:.4f}")

print("\n" + "=" * 80)
print("PROBLEM IDENTIFIED:")
print("=" * 80)

if confidences.max() < 0.70:
    print("❌ Model is NEVER confident enough (max < 70%)")
    print("\nReasons:")
    print("  1. Model is overfitting (Train AUROC 0.76 vs Val AUROC 0.54)")
    print("  2. LightGBM regularization too weak")
    print("  3. Too many features (87) causing noise")
    
    print("\nSOLUTIONS:")
    print("  1. Lower threshold to 0.55-0.60")
    print("  2. Use fewer features (top 20-30)")
    print("  3. More regularization (lower learning_rate, higher min_child_samples)")
    print("  4. More data (60-90 days)")
elif confidences.max() < 0.75:
    print("⚠️  Model reaches 70% but not 75%")
    print(f"   Use confidence threshold: 0.65-0.70")
else:
    print("✅ Model is confident enough")
    print(f"   Profitable at 0.75 threshold")

