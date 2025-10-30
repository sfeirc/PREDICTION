"""
Fetch MAXIMUM historical data WITHOUT downsampling for MAXIMUM TRADES!
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from data_fetcher import BinanceDataFetcher
from feature_engineering_v2 import FeatureEngineV2
import yaml

print("=" * 80)
print("📥 FETCHING MAXIMUM DATA (NO DOWNSAMPLING)")
print("=" * 80)

with open("config_profitable.yaml") as f:
    config = yaml.safe_load(f)

# DISABLE DOWNSAMPLING TO KEEP ALL DATA!
config["target"]["balance_classes"] = False

print("\n🎯 Strategy: Keep ALL samples (no downsampling)")
print("   This maximizes trading opportunities!\n")

fetcher = BinanceDataFetcher("data/raw")

# BTC
print("📊 Fetching BTCUSDT...")
df_btc = fetcher.fetch_and_cache("BTCUSDT", days=90, interval="1m", refresh=False)
print(f"   ✓ Got {len(df_btc):,} candles")
print(f"   Date range: {df_btc.index.min()} to {df_btc.index.max()}")

# ETH for cross-asset features
print("\n📊 Fetching ETHUSDT...")
df_eth = fetcher.fetch_and_cache("ETHUSDT", days=90, interval="1m", refresh=False)
print(f"   ✓ Got {len(df_eth):,} candles")

# Now create features WITHOUT downsampling
print("\n" + "=" * 80)
print("🔧 CREATING FEATURES (ALL DATA)")
print("=" * 80)

fe = FeatureEngineV2(config)
result = fe.create_features(df_btc, df_eth)

# Create targets but DON'T balance
result = fe.create_multi_horizon_targets(
    result,
    horizons=config["target"]["horizons"],
    up_threshold=config["target"]["up_threshold"],
    down_threshold=config["target"]["down_threshold"],
)

# Remove NaN targets only
result = result[~result["target_5m"].isna()].copy()

print(f"\n✅ Features created: {len(result)} samples (NO DOWNSAMPLING!)")

# Save
result.to_parquet("data/processed/btcusdt_features_v2_full.parquet")
print(f"   Saved to: data/processed/btcusdt_features_v2_full.parquet")

print("\n🚀 This dataset has ~10X MORE SAMPLES!")
print("   = MORE TRADING OPPORTUNITIES!")
print("   = MORE PROFIT POTENTIAL!")

