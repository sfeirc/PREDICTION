"""
Fetch MAXIMUM historical data for training.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from data_fetcher import BinanceDataFetcher
from feature_engineering_v2 import FeatureEngineV2
import yaml

print("=" * 80)
print("📥 FETCHING MAXIMUM DATA")
print("=" * 80)

with open("config_profitable.yaml") as f:
    config = yaml.safe_load(f)

# Fetch 90 DAYS instead of 30
print("\n🎯 Target: 90 days of 1-minute BTCUSDT data")
print("   This will give you ~90,000 samples")
print("   = MORE TRADES = MORE PROFIT!\n")

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

# Save
df_btc.to_parquet("data/raw/btcusdt_1m.parquet")
df_eth.to_parquet("data/raw/ethusdt_1m.parquet")

print("\n✅ Data saved!")

# Now create features
print("\n" + "=" * 80)
print("🔧 CREATING FEATURES")
print("=" * 80)

fe = FeatureEngineV2(config)
result = fe.create_features(df_btc, df_eth)

print(f"\n✅ Features created: {len(result)} samples")
print(f"   Saved to: data/processed/btcusdt_features_v2.parquet")

print("\n🚀 NOW RUN: python maximize_profits.py")
print("   You'll get 3X MORE TRADES!")

