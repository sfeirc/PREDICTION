"""
Analyze why predictions are low and show recent predictions
"""

import sys
import io
import yaml
import numpy as np
import pandas as pd
from data_manager_worldclass import DataManagerWorldClass
from feature_engine_worldclass import FeatureEngineWorldClass
from lightgbm import LGBMClassifier
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("="*80)
print("📊 ANALYZING PREDICTIONS - Recent History")
print("="*80)

# Load config
with open('config_ultimate.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f) or {}
config.setdefault('data', {})
config['data'].setdefault('correlation_pairs', ['ETHUSDT', 'BNBUSDT'])

# Fetch and prepare data
print("\n1. Loading data...")
dm = DataManagerWorldClass(config)
data = dm.fetch_all_data()
fe = FeatureEngineWorldClass(config)
df = fe.create_features(data)
df = df.replace([np.inf, -np.inf], np.nan).dropna()

feature_cols = [c for c in df.columns if not c.startswith('target_')]
target_cols = [c for c in df.columns if c.startswith('target_')]
target_col = target_cols[0]

# Train model
print("2. Training model...")
split_idx = int(len(df) * 0.8)
df_train = df.iloc[:split_idx]

X_train = df_train[feature_cols].values
y_train = df_train[target_col].values

# Load best params
with open('logs/best_settings.yaml', 'r', encoding='utf-8') as f:
    best = yaml.safe_load(f) or {}
best_params = best.get('best_lightgbm_params', {})

model = LGBMClassifier(
    n_estimators=300,
    learning_rate=float(best_params.get('learning_rate', 0.03)),
    max_depth=int(best_params.get('max_depth', 8)),
    num_leaves=int(best_params.get('num_leaves', 64)),
    min_child_samples=int(best_params.get('min_child_samples', 20)),
    subsample=float(best_params.get('subsample', 0.8)),
    colsample_bytree=float(best_params.get('colsample_bytree', 0.8)),
    random_state=42,
    verbose=-1
)

model.fit(X_train, y_train)

# Analyze last 100 rows
print("\n3. Analyzing last 100 predictions...")
df_test = df.iloc[-100:].copy()
X_test = df_test[feature_cols].fillna(0.0).values

predictions = model.predict_proba(X_test)[:, 1]
df_test['prediction'] = predictions

# Add signal info
df_test['signal'] = 'HOLD'
df_test.loc[df_test['prediction'] > 0.6, 'signal'] = 'BUY'
df_test.loc[df_test['prediction'] < 0.4, 'signal'] = 'SELL'

# Show statistics
print(f"\n   Predictions range: [{predictions.min():.4f}, {predictions.max():.4f}]")
print(f"   Predictions mean: {predictions.mean():.4f}")
print(f"   BUY signals (>0.6): {(predictions > 0.6).sum()}")
print(f"   SELL signals (<0.4): {(predictions < 0.4).sum()}")
print(f"   HOLD signals [0.4-0.6]: {((predictions >= 0.4) & (predictions <= 0.6)).sum()}")

# Show last 20 predictions
print("\n4. Last 20 predictions:")
print("-"*80)
results = df_test[['close', 'prediction', 'signal']].tail(20).copy()
if isinstance(results.index, pd.DatetimeIndex):
    results.index = results.index.strftime('%Y-%m-%d %H:%M')

for idx, row in results.iterrows():
    price_str = f"${row['close']:,.2f}" if pd.notna(row['close']) else "N/A"
    pred_str = f"{row['prediction']:.4f}"
    signal_str = row['signal']
    
    # Color coding
    if signal_str == 'BUY':
        signal_emoji = "🟢"
    elif signal_str == 'SELL':
        signal_emoji = "🔴"
    else:
        signal_emoji = "⏸️"
    
    print(f"   {idx:20s} | Price: {price_str:>12s} | Pred: {pred_str:>7s} | {signal_emoji} {signal_str}")

# Show latest prediction details
print("\n5. Latest prediction details:")
print("-"*80)
latest = df_test.iloc[-1]
print(f"   Timestamp: {df_test.index[-1]}")
print(f"   Price: ${latest['close']:,.2f}")
print(f"   Prediction: {latest['prediction']:.4f}")
print(f"   Signal: {latest['signal']}")
print(f"   Confidence: {abs(latest['prediction'] - 0.5) * 2:.2%}")

if latest['prediction'] < 0.01:
    print(f"\n   ⚠️  WARNING: Prediction is very low ({latest['prediction']:.4f})")
    print(f"   This suggests the model strongly predicts DOWN movement")
    print(f"   Reasons could be:")
    print(f"   - Recent price action indicates downward trend")
    print(f"   - Features are showing bearish signals")
    print(f"   - Market conditions are unfavorable")

# Feature importance check
print("\n6. Top 10 feature values (latest row):")
print("-"*80)
latest_features = df_test[feature_cols].iloc[-1].abs().sort_values(ascending=False).head(10)
for feat, val in latest_features.items():
    print(f"   {feat:30s}: {val:12.4f}")

print("\n" + "="*80)
print("✅ Analysis complete!")
print("="*80)
print("\n💡 RECOMMENDATION:")
if (predictions > 0.6).sum() > 0 or (predictions < 0.4).sum() > 0:
    print("   Model IS generating signals - check recent predictions above")
    print("   If no trades, it might be due to timing or position management")
else:
    print("   Model predictions are mostly in middle range (0.4-0.6)")
    print("   This means low confidence - consider:")
    print("   - Lowering confidence threshold (currently 0.6)")
    print("   - Training on more recent data")
    print("   - Checking if market is in consolidation phase")

