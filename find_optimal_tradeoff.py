"""
🎯 FIND OPTIMAL TRADE-OFF: Maximum Trades + Profitability

Test different thresholds to find sweet spot.
"""

import sys
import io
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from datasets import TimeSeriesDataset
from models.baselines_v2 import BaselineModelV2
from torch.utils.data import DataLoader
from utils import set_seed
from maximize_profits import MaxProfitTrader


def test_threshold(threshold_pct, config, df_btc, df_eth):
    """Test a specific threshold and return results."""
    from feature_engineering_v2 import FeatureEngineV2
    
    threshold = threshold_pct / 100  # Convert to decimal
    
    # Create features with this threshold
    config_copy = config.copy()
    config_copy["target"]["up_threshold"] = threshold
    config_copy["target"]["down_threshold"] = -threshold
    config_copy["target"]["balance_classes"] = False
    
    fe = FeatureEngineV2(config_copy)
    df = fe.create_features(df_btc, df_eth)
    df = fe.create_multi_horizon_targets(
        df,
        horizons=[5],
        up_threshold=threshold,
        down_threshold=-threshold,
    )
    df = df[~df["target_5m"].isna()].copy()
    
    # Get features
    exclude_cols = {"open", "high", "low", "close", "volume", "quote_volume", "trades", 
                   "taker_buy_base", "taker_buy_quote", "target_5m", "target_10m", "target_15m",
                   "forward_return_5m", "forward_return_10m", "forward_return_15m",
                   "hour", "minute", "day_of_week", "mid_price", "sample_weight", "regime", "rolling_vol"}
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Split
    split_idx = int(len(df) * 0.85)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    # Create datasets
    train_dataset = TimeSeriesDataset(train_df, feature_cols, "target_5m", config["sequence"]["lookback_minutes"])
    test_dataset = TimeSeriesDataset(test_df, feature_cols, "target_5m", config["sequence"]["lookback_minutes"], train_dataset.scaler)
    
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=False)
    
    # Train
    model = BaselineModelV2("lightgbm", config_copy)
    model.train(train_loader, None)
    
    # Predict
    X_test = test_dataset.feature_values
    y_pred, y_pred_proba = model.predict(X_test)
    
    test_indices = test_df.index[config["sequence"]["lookback_minutes"]:][:len(y_pred)]
    actual_returns = test_df.loc[test_indices, "forward_return_5m"].values
    
    valid_mask = ~np.isnan(actual_returns)
    y_pred = y_pred[valid_mask]
    y_pred_proba = y_pred_proba[valid_mask]
    actual_returns = actual_returns[valid_mask]
    
    # Test trading at 70% confidence
    trader = MaxProfitTrader(initial_capital=10000, trading_cost=0.001)
    result = trader.simulate_with_compounding(
        y_pred, y_pred_proba, actual_returns, 0.70, use_kelly=False
    )
    
    return {
        'threshold_pct': threshold_pct,
        'samples': len(df),
        'trades': result['n_trades'],
        'win_rate': result['win_rate'],
        'return': result['total_return'],
        'capital': result['final_capital'],
    }


def main():
    print("=" * 80)
    print("🎯 FINDING OPTIMAL TRADE-OFF")
    print("=" * 80)
    print("\nTesting different thresholds to maximize:")
    print("  • Number of trades (more opportunities)")
    print("  • Profitability (positive returns)")
    
    # Load config
    with open("config_profitable.yaml") as f:
        config = yaml.safe_load(f)
    
    set_seed(config["seed"])
    
    # Get raw data
    from data_fetcher import BinanceDataFetcher
    
    print("\n📥 Loading data...")
    fetcher = BinanceDataFetcher("data/raw")
    df_btc = fetcher.fetch_and_cache("BTCUSDT", days=90, interval="1m", refresh=False)
    df_eth = fetcher.fetch_and_cache("ETHUSDT", days=90, interval="1m", refresh=False)
    print(f"   ✓ Loaded {len(df_btc):,} candles")
    
    # Test different thresholds
    thresholds = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]  # Percentages
    
    all_results = []
    
    for thresh in thresholds:
        print(f"\n{'='*80}")
        print(f"🔍 Testing threshold: {thresh}%")
        print(f"{'='*80}")
        
        result = test_threshold(thresh, config, df_btc, df_eth)
        all_results.append(result)
        
        print(f"\n   Samples: {result['samples']:,}")
        print(f"   Trades: {result['trades']}")
        print(f"   Win Rate: {result['win_rate']:.1%}")
        print(f"   Return: {result['return']:+.2f}%")
        print(f"   Final Capital: ${result['capital']:,.2f}")
    
    # Summary
    results_df = pd.DataFrame(all_results)
    
    print("\n" + "=" * 80)
    print("📊 COMPLETE COMPARISON")
    print("=" * 80)
    
    print(f"\n{'Threshold':<12} {'Samples':<10} {'Trades':<10} {'Win Rate':<12} {'Return':<12}")
    print("-" * 80)
    for _, row in results_df.iterrows():
        print(f"{row['threshold_pct']:<11.2f}% "
              f"{int(row['samples']):<10d} "
              f"{int(row['trades']):<10d} "
              f"{row['win_rate']:<12.1%} "
              f"{row['return']:+11.2f}%")
    
    # Find best profitable option
    profitable = results_df[results_df['return'] > 0]
    
    if len(profitable) > 0:
        # Maximize trades among profitable
        best = profitable.loc[profitable['trades'].idxmax()]
        
        print("\n" + "=" * 80)
        print("🏆 OPTIMAL CONFIGURATION")
        print("=" * 80)
        
        print(f"\n✅ BEST THRESHOLD: {best['threshold_pct']}%")
        print(f"\n   📊 Dataset Size: {int(best['samples']):,} samples")
        print(f"   🎯 Trades: {int(best['trades'])} trades (at 70% confidence)")
        print(f"   🎲 Win Rate: {best['win_rate']:.1%}")
        print(f"   📈 Return: {best['return']:+.2f}%")
        print(f"   💰 Final Capital: ${best['capital']:,.2f}")
        print(f"   💵 Profit: ${best['capital'] - 10000:,.2f}")
        
        print(f"\n💡 TO USE THIS CONFIGURATION:")
        print(f"   1. Set up_threshold: {best['threshold_pct']/100:.4f} in config")
        print(f"   2. Set down_threshold: {-best['threshold_pct']/100:.4f} in config")
        print(f"   3. Use min_confidence: 0.70")
        print(f"   4. Disable balance_classes (or use weights)")
        
        # Scale-up projection
        print(f"\n📈 SCALING PROJECTIONS:")
        if best['trades'] > 0:
            profit_per_trade = (best['capital'] - 10000) / best['trades']
            print(f"\n   Average profit per trade: ${profit_per_trade:.2f}")
            print(f"\n   With this strategy running continuously:")
            print(f"      If 100 trades/month → ${profit_per_trade * 100:,.2f}/month")
            print(f"      If 500 trades/month → ${profit_per_trade * 500:,.2f}/month")
        
    else:
        print("\n⚠️  No profitable threshold found!")
        print("\n   All tested thresholds are unprofitable.")
        print("   The model needs improvement in:")
        print("   - Feature engineering")
        print("   - Model architecture")
        print("   - More training data")
        print("   - Better risk management")
    
    # Save results
    results_df.to_csv("logs/threshold_optimization.csv", index=False)
    print(f"\n💾 Results saved to: logs/threshold_optimization.csv")
    
    print("\n" + "=" * 80)
    print("🎉 OPTIMIZATION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()

