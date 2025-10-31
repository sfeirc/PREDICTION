"""
Quick diagnostic to check why predictions are 0
"""

import sys
import io
import yaml
import numpy as np
import pandas as pd
from data_manager_worldclass import DataManagerWorldClass
from feature_engine_worldclass import FeatureEngineWorldClass
from lightgbm import LGBMClassifier

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("="*80)
print("🔍 QUICK DIAGNOSTIC - Why predictions might be 0")
print("="*80)

# Load config
with open('config_ultimate.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f) or {}
config.setdefault('data', {})
config['data'].setdefault('correlation_pairs', ['ETHUSDT', 'BNBUSDT'])

# Fetch data
print("\n1. Fetching data...")
dm = DataManagerWorldClass(config)
data = dm.fetch_all_data()

# Create features
print("2. Creating features...")
fe = FeatureEngineWorldClass(config)
df = fe.create_features(data)
df = df.replace([np.inf, -np.inf], np.nan).dropna()

print(f"   Dataset shape: {df.shape}")

# Identify features and target
feature_cols = [c for c in df.columns if not c.startswith('target_')]
target_cols = [c for c in df.columns if c.startswith('target_')]

print(f"   Features: {len(feature_cols)}")
print(f"   Targets: {len(target_cols)}")

if not target_cols:
    print("❌ ERROR: No target columns!")
    sys.exit(1)

target_col = target_cols[0]
print(f"   Using target: {target_col}")

# Split data
print("\n3. Training model...")
split_idx = int(len(df) * 0.8)
df_train = df.iloc[:split_idx]
df_test = df.iloc[split_idx:]

X_train = df_train[feature_cols].values
y_train = df_train[target_col].values
X_test = df_test[feature_cols].values
y_test = df_test[target_col].values

print(f"   Train samples: {len(X_train)}")
print(f"   Test samples: {len(X_test)}")

# Load best params
try:
    with open('logs/best_settings.yaml', 'r', encoding='utf-8') as f:
        best = yaml.safe_load(f) or {}
    best_params = best.get('best_lightgbm_params', {})
    print(f"   Using optimized params")
except:
    best_params = {}
    print(f"   Using default params")

# Train
model = LGBMClassifier(
    n_estimators=100,  # Smaller for quick test
    learning_rate=float(best_params.get('learning_rate', 0.03)),
    max_depth=int(best_params.get('max_depth', 8)),
    num_leaves=int(best_params.get('num_leaves', 64)),
    random_state=42,
    verbose=-1
)

model.fit(X_train, y_train)

# Test predictions
print("\n4. Testing predictions...")
y_pred_proba = model.predict_proba(X_test)[:, 1]

print(f"   Predictions shape: {y_pred_proba.shape}")
print(f"   Prediction range: [{y_pred_proba.min():.4f}, {y_pred_proba.max():.4f}]")
print(f"   Prediction mean: {y_pred_proba.mean():.4f}")
print(f"   Predictions > 0.6: {(y_pred_proba > 0.6).sum()} / {len(y_pred_proba)}")
print(f"   Predictions < 0.4: {(y_pred_proba < 0.4).sum()} / {len(y_pred_proba)}")
print(f"   Predictions in [0.4, 0.6]: {((y_pred_proba >= 0.4) & (y_pred_proba <= 0.6)).sum()} / {len(y_pred_proba)}")

# Test on latest row
print("\n5. Testing on latest row (simulating live prediction)...")
df_latest = df.iloc[[-1]]
X_latest = df_latest[feature_cols].fillna(0.0).values

proba = model.predict_proba(X_latest)[:, 1][0]
print(f"   Latest prediction: {proba:.4f}")
print(f"   Would BUY: {proba > 0.6}")
print(f"   Would SELL: {proba < 0.4}")

# Check feature values
print("\n6. Feature statistics (latest row)...")
latest_features = pd.Series(df_latest[feature_cols].iloc[0])
print(f"   NaN features: {latest_features.isna().sum()} / {len(latest_features)}")
print(f"   Inf features: {np.isinf(latest_features).sum()} / {len(latest_features)}")
print(f"   Zero features: {(latest_features == 0).sum()} / {len(latest_features)}")
print(f"   Non-zero features: {(latest_features != 0).sum()} / {len(latest_features)}")

print("\n" + "="*80)
print("✅ Diagnostic complete!")
print("="*80)

