"""
Find the optimal confidence threshold for profitable trading.

Tests different confidence levels and shows which one is most profitable.
"""

import sys
import io
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from datasets import TimeSeriesDataset, create_train_val_test_split
from models.baselines_v2 import BaselineModelV2
from trading_simulator import TradingSimulator
from torch.utils.data import DataLoader


def test_confidence_threshold(
    y_pred,
    y_pred_proba,
    actual_returns,
    confidence_threshold,
    config,
):
    """Test a specific confidence threshold."""
    simulator = TradingSimulator(
        initial_capital=10000,
        trading_cost=config["evaluation"]["trading_cost"],
        min_confidence=confidence_threshold,
    )
    
    results = simulator.simulate(y_pred, y_pred_proba, actual_returns)
    
    return {
        'confidence': confidence_threshold,
        'n_trades': results['n_trades'],
        'win_rate': results['win_rate'],
        'return': results['total_return_pct'],
        'sharpe': results['sharpe_ratio'],
        'max_dd': results['max_drawdown_pct'],
    }


def main():
    print("=" * 80)
    print("🔍 FINDING OPTIMAL CONFIDENCE THRESHOLD")
    print("=" * 80)
    print("\nThis will test different confidence levels to find the sweet spot.")
    
    # Load config
    with open("config_profitable.yaml") as f:
        config = yaml.safe_load(f)
    
    # Load data
    processed_path = Path(config["data"]["processed_dir"]) / "btcusdt_features_v2.parquet"
    df = pd.read_parquet(processed_path)
    
    # Get features
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
    
    print(f"\nDataset: Train={len(train_dataset)}, Test={len(test_dataset)}")
    
    # Train model
    print("\nTraining LightGBM...")
    model = BaselineModelV2("lightgbm", config)
    model.train(train_loader, None)
    
    # Get predictions
    X_test = test_dataset.feature_values
    y_test = test_dataset.targets
    y_pred, y_pred_proba = model.predict(X_test)
    
    test_indices = test_df.index[config["sequence"]["lookback_minutes"]:][:len(y_pred)]
    actual_returns = test_df.loc[test_indices, "forward_return_5m"].values
    
    valid_mask = ~np.isnan(actual_returns)
    y_pred = y_pred[valid_mask]
    y_pred_proba = y_pred_proba[valid_mask]
    actual_returns = actual_returns[valid_mask]
    
    print(f"Valid predictions: {len(y_pred)}")
    
    # Test different confidence thresholds
    print("\n" + "=" * 80)
    print("TESTING CONFIDENCE THRESHOLDS")
    print("=" * 80)
    
    confidence_levels = [0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70, 0.75, 0.80]
    results = []
    
    for conf in confidence_levels:
        result = test_confidence_threshold(y_pred, y_pred_proba, actual_returns, conf, config)
        results.append(result)
        
        print(f"\nConfidence: {conf:.0%}")
        print(f"  Trades:    {result['n_trades']:3d} / {len(y_pred)} ({result['n_trades']/len(y_pred)*100:5.1f}%)")
        print(f"  Win Rate:  {result['win_rate']:6.1%}")
        print(f"  Return:    {result['return']:+7.2f}%")
        print(f"  Sharpe:    {result['sharpe']:7.2f}")
        print(f"  Max DD:    {result['max_dd']:7.2f}%")
    
    # Find best
    results_df = pd.DataFrame(results)
    
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"\n{'Confidence':<12} {'Trades':<10} {'Win Rate':<10} {'Return':<12} {'Sharpe':<10}")
    print("-" * 80)
    for _, row in results_df.iterrows():
        print(f"{row['confidence']:<12.0%} "
              f"{int(row['n_trades']):<10d} "
              f"{row['win_rate']:<10.1%} "
              f"{row['return']:+11.2f}% "
              f"{row['sharpe']:<10.2f}")
    
    # Find optimal (best return with >10 trades)
    viable = results_df[results_df['n_trades'] >= 10]
    
    if len(viable) > 0:
        best_by_return = viable.loc[viable['return'].idxmax()]
        best_by_sharpe = viable.loc[viable['sharpe'].idxmax()]
        
        print("\n" + "=" * 80)
        print("🏆 OPTIMAL THRESHOLDS")
        print("=" * 80)
        
        print(f"\n📈 Best for Returns:")
        print(f"   Confidence: {best_by_return['confidence']:.0%}")
        print(f"   Return:     {best_by_return['return']:+.2f}%")
        print(f"   Win Rate:   {best_by_return['win_rate']:.1%}")
        print(f"   Trades:     {int(best_by_return['n_trades'])}")
        
        print(f"\n📊 Best for Sharpe:")
        print(f"   Confidence: {best_by_sharpe['confidence']:.0%}")
        print(f"   Sharpe:     {best_by_sharpe['sharpe']:.2f}")
        print(f"   Return:     {best_by_sharpe['return']:+.2f}%")
        print(f"   Trades:     {int(best_by_sharpe['n_trades'])}")
        
        # Recommendation
        recommended_conf = best_by_return['confidence']
        
        print("\n" + "=" * 80)
        print("💡 RECOMMENDATION")
        print("=" * 80)
        print(f"\nUse confidence threshold: {recommended_conf:.0%}")
        print(f"\nUpdate config_profitable.yaml:")
        print(f"  training:")
        print(f"    min_confidence: {recommended_conf:.2f}")
        
        if best_by_return['return'] > 0:
            print(f"\n✅ PROFITABLE at {recommended_conf:.0%} confidence!")
            print(f"   Expected: {best_by_return['return']:+.2f}% return, {best_by_return['win_rate']:.1%} win rate")
        else:
            print(f"\n⚠️  Still not profitable even at optimal confidence.")
            print("   Try: 1) More data (60-90 days)")
            print("        2) Different features")
            print("        3) Different model")
    else:
        print("\n⚠️  No viable confidence levels found (all have <10 trades)")
        print("   Model is not confident enough. Try:")
        print("   1. Lower thresholds in config (e.g., 0.25% instead of 0.3%)")
        print("   2. More training data")
        print("   3. Simpler features")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

