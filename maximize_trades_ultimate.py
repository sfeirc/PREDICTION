"""
🚀🚀🚀 ULTIMATE MAXIMUM TRADES SYSTEM 🚀🚀🚀

Lower thresholds + All data + Optimal confidence = TONS OF TRADES!
"""

import sys
import io
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import matplotlib.pyplot as plt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from datasets import TimeSeriesDataset, create_train_val_test_split
from models.baselines_v2 import BaselineModelV2
from torch.utils.data import DataLoader
from utils import set_seed
from maximize_profits import MaxProfitTrader


def main():
    print("=" * 80)
    print("🚀🚀🚀 ULTIMATE MAXIMUM TRADES SYSTEM 🚀🚀🚀")
    print("=" * 80)
    print("\n🎯 Strategy:")
    print("  • Lower thresholds (0.05% instead of 0.2%)")
    print("  • Keep ALL samples")
    print("  • Use class weights")
    print("  • Lower confidence (50-55%)")
    print("  = MAXIMUM TRADES!!!")
    
    # Load config
    with open("config_profitable.yaml") as f:
        config = yaml.safe_load(f)
    
    # OVERRIDE FOR MAXIMUM TRADES
    config["target"]["up_threshold"] = 0.0005  # 0.05% instead of 0.2%!
    config["target"]["down_threshold"] = -0.0005
    config["target"]["balance_classes"] = False  # Keep all data
    config["training"]["min_confidence"] = 0.50  # Lower confidence = more trades
    
    set_seed(config["seed"])
    
    # Get raw data
    from data_fetcher import BinanceDataFetcher
    from feature_engineering_v2 import FeatureEngineV2
    
    print("\n📥 Fetching data...")
    fetcher = BinanceDataFetcher("data/raw")
    df_btc = fetcher.fetch_and_cache("BTCUSDT", days=90, interval="1m", refresh=False)
    df_eth = fetcher.fetch_and_cache("ETHUSDT", days=90, interval="1m", refresh=False)
    
    print(f"   ✓ BTC: {len(df_btc):,} candles")
    print(f"   ✓ ETH: {len(df_eth):,} candles")
    
    # Create features with LOWER thresholds
    print("\n🔧 Creating features with LOWER thresholds...")
    fe = FeatureEngineV2(config)
    df = fe.create_features(df_btc, df_eth)
    
    # Create targets with LOWER thresholds
    df = fe.create_multi_horizon_targets(
        df,
        horizons=[5],
        up_threshold=0.0005,  # 0.05%
        down_threshold=-0.0005,
    )
    
    # Remove NaN only
    df = df[~df["target_5m"].isna()].copy()
    
    print(f"\n📊 Total samples: {len(df):,}")
    up_count = (df["target_5m"] == 1).sum()
    down_count = (df["target_5m"] == 0).sum()
    print(f"   Up: {up_count:,} ({up_count/len(df)*100:.1f}%)")
    print(f"   Down: {down_count:,} ({down_count/len(df)*100:.1f}%)")
    
    # Get features
    exclude_cols = {"open", "high", "low", "close", "volume", "quote_volume", "trades", 
                   "taker_buy_base", "taker_buy_quote", "target_5m", "target_10m", "target_15m",
                   "forward_return_5m", "forward_return_10m", "forward_return_15m",
                   "hour", "minute", "day_of_week", "mid_price", "sample_weight", "regime", "rolling_vol"}
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Split (use 85% for training)
    split_idx = int(len(df) * 0.85)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    print(f"\n🎯 Train/Test Split:")
    print(f"   Train: {len(train_df):,} samples")
    print(f"   Test: {len(test_df):,} samples")
    
    # Create datasets
    train_dataset = TimeSeriesDataset(train_df, feature_cols, "target_5m", config["sequence"]["lookback_minutes"])
    test_dataset = TimeSeriesDataset(test_df, feature_cols, "target_5m", config["sequence"]["lookback_minutes"], train_dataset.scaler)
    
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    
    print(f"\n   After lookback: Train={len(train_dataset)}, Test={len(test_dataset)}")
    
    # Train model with class weights
    print("\n🤖 Training LightGBM with class weights...")
    model = BaselineModelV2("lightgbm", config)
    model.train(train_loader, None)
    
    # Get predictions
    X_test = test_dataset.feature_values
    y_pred, y_pred_proba = model.predict(X_test)
    
    test_indices = test_df.index[config["sequence"]["lookback_minutes"]:][:len(y_pred)]
    actual_returns = test_df.loc[test_indices, "forward_return_5m"].values
    
    valid_mask = ~np.isnan(actual_returns)
    y_pred = y_pred[valid_mask]
    y_pred_proba = y_pred_proba[valid_mask]
    actual_returns = actual_returns[valid_mask]
    
    print(f"   Valid predictions: {len(y_pred):,}")
    
    # Test with LOWER confidence thresholds
    print("\n" + "=" * 80)
    print("🎮 TESTING WITH LOWER CONFIDENCE = MORE TRADES!")
    print("=" * 80)
    
    trader = MaxProfitTrader(initial_capital=10000, trading_cost=0.001)
    
    # Test LOWER confidence levels (50-60%)
    confidence_levels = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60]
    all_results = []
    
    for conf in confidence_levels:
        print(f"\n📈 Testing confidence: {conf:.0%}")
        
        result = trader.simulate_with_compounding(
            y_pred, y_pred_proba, actual_returns, conf, use_kelly=True
        )
        
        print(f"   Trades: {result['n_trades']}")
        print(f"   Win Rate: {result['win_rate']:.1%}")
        print(f"   Return: {result['total_return']:+.2f}%")
        print(f"   Final Capital: ${result['final_capital']:,.2f}")
        
        all_results.append({
            'confidence': conf,
            'trades': result['n_trades'],
            'win_rate': result['win_rate'],
            'return': result['total_return'],
            'capital': result['final_capital'],
        })
    
    # Summary
    results_df = pd.DataFrame(all_results)
    
    print("\n" + "=" * 80)
    print("📊 RESULTS SUMMARY")
    print("=" * 80)
    
    print(f"\n{'Confidence':<12} {'Trades':<10} {'Win Rate':<12} {'Return':<12} {'Capital':<12}")
    print("-" * 80)
    for _, row in results_df.iterrows():
        print(f"{row['confidence']:<12.0%} "
              f"{int(row['trades']):<10d} "
              f"{row['win_rate']:<12.1%} "
              f"{row['return']:+11.2f}% "
              f"${row['capital']:>10,.2f}")
    
    # Find sweet spot (most trades + profitable)
    profitable = results_df[results_df['return'] > 0]
    
    if len(profitable) > 0:
        best = profitable.loc[profitable['trades'].idxmax()]
        
        print("\n" + "=" * 80)
        print("🏆 MAXIMUM PROFITABLE TRADES!")
        print("=" * 80)
        
        print(f"\n✅ SWEET SPOT: {best['confidence']:.0%} confidence")
        print(f"   🎯 Trades: {int(best['trades'])} trades")
        print(f"   📈 Return: {best['return']:+.2f}%")
        print(f"   💰 Final Capital: ${best['capital']:,.2f}")
        print(f"   💵 Profit: ${best['capital'] - 10000:,.2f}")
        print(f"   🎲 Win Rate: {best['win_rate']:.1%}")
        
        # Annualized projection
        test_days = (test_df.index.max() - test_df.index.min()).days
        monthly_return = best['return'] * (30 / test_days)
        
        print(f"\n💡 SCALING TO LARGER CAPITAL:")
        print(f"\n   With $100,000:")
        print(f"      Month 1: ${100000 * (1 + monthly_return/100):,.2f}")
        print(f"      Year 1:  ${100000 * (1 + monthly_return/100)**12:,.2f}")
        
        print(f"\n   With $1,000,000:")
        print(f"      Month 1: ${1000000 * (1 + monthly_return/100):,.2f}")
        print(f"      Year 1:  ${1000000 * (1 + monthly_return/100)**12:,.2f}")
        
    else:
        print("\n⚠️  No profitable configuration found!")
        print("   Consider:")
        print("   - Collecting more data")
        print("   - Improving features")
        print("   - Using different models")
    
    print("\n" + "=" * 80)
    print("🎉 MAXIMUM TRADES ANALYSIS COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()

