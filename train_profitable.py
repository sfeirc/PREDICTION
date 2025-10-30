"""
PROFIT-OPTIMIZED Training Script

Key improvements based on research:
1. Stricter thresholds (0.3% vs 0.2%)
2. Higher confidence filtering (0.7 vs 0.6)
3. Trade only in favorable conditions
4. Risk management (stop loss, take profit)
5. Feature selection (top features only)
"""

import sys
import io
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from data_fetcher import BinanceDataFetcher
from feature_engineering_v2 import FeatureEngineV2
from datasets import TimeSeriesDataset, create_train_val_test_split
from models.baselines_v2 import BaselineModelV2
from utils import set_seed
from trading_simulator import TradingSimulator
from torch.utils.data import DataLoader


class ProfitOptimizedTrainer:
    """Training with profit optimization."""
    
    def __init__(self, config):
        self.config = config
        
    def select_top_features(self, model, feature_cols, X_train, y_train, top_n=30):
        """Select only the most important features."""
        print(f"\n  Selecting top {top_n} features...")
        
        # Train quick model to get feature importance
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        # Get importances
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        top_features = [feature_cols[i] for i in indices]
        
        print(f"  Top {top_n} features selected:")
        for i, (idx, feat) in enumerate(zip(indices[:10], top_features[:10]), 1):
            print(f"    {i:2d}. {feat:40s}: {importances[idx]:.4f}")
        
        return top_features
    
    def filter_by_regime(self, df, regime='high'):
        """Filter to only trade in specific volatility regime."""
        if 'vol_regime_high' in df.columns and regime == 'high':
            return df[df['vol_regime_high'] == 1].copy()
        return df
    
    def filter_by_trading_hours(self, df, allowed_hours):
        """Filter to only trade during high-liquidity hours."""
        if 'hour' in df.columns:
            return df[df['hour'].isin(allowed_hours)].copy()
        return df
    
    def apply_risk_management(self, y_pred, y_pred_proba, actual_returns):
        """Apply stop-loss and take-profit rules."""
        stop_loss = self.config['evaluation'].get('stop_loss', 0.01)
        take_profit = self.config['evaluation'].get('take_profit', 0.02)
        
        # Simulate with risk management
        adjusted_returns = actual_returns.copy()
        
        for i in range(len(adjusted_returns)):
            if y_pred[i] == 1:  # Long position
                # Apply stop loss
                if adjusted_returns[i] < -stop_loss:
                    adjusted_returns[i] = -stop_loss
                # Apply take profit
                elif adjusted_returns[i] > take_profit:
                    adjusted_returns[i] = take_profit
        
        return adjusted_returns


def main():
    parser = argparse.ArgumentParser(description="Profit-optimized training")
    parser.add_argument("--config", default="config_profitable.yaml")
    parser.add_argument("--model", default="lightgbm")
    args = parser.parse_args()
    
    print("=" * 80)
    print("💰 PROFIT-OPTIMIZED TRAINING")
    print("=" * 80)
    print("\nKey optimizations:")
    print("  ✓ Stricter thresholds (0.3% vs 0.2%)")
    print("  ✓ Higher confidence (70% vs 60%)")
    print("  ✓ Risk management (stop-loss, take-profit)")
    print("  ✓ Feature selection (top 30 features)")
    print("  ✓ Trade only in favorable conditions")
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    set_seed(config["seed"])
    
    # Check data requirements
    min_days = 30
    if config["data"]["days"] < min_days:
        print(f"\n⚠️  WARNING: Using {config['data']['days']} days of data")
        print(f"   For profitable trading, recommended: {min_days}+ days")
        print(f"\n   Run: python data_fetcher.py --symbol BTCUSDT --days {min_days}")
        response = input("\n   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Load or create features
    symbol = config["data"]["train_symbol"]
    processed_path = Path(config["data"]["processed_dir"]) / f"{symbol.lower()}_features_v2.parquet"
    
    if not processed_path.exists():
        print("\n[1/5] Creating enhanced features...")
        
        fetcher = BinanceDataFetcher(cache_dir=config["data"]["cache_dir"])
        
        btc_df = fetcher.fetch_and_cache(
            symbol="BTCUSDT",
            days=config["data"]["days"],
            interval="1m",
        )
        
        eth_df = fetcher.fetch_and_cache(
            symbol="ETHUSDT",
            days=config["data"]["days"],
            interval="1m",
        ) if config["features"]["use_cross_asset"] else None
        
        feature_engine = FeatureEngineV2(config)
        df = feature_engine.create_features(btc_df, cross_asset_df=eth_df)
        df = feature_engine.create_multi_horizon_targets(
            df,
            horizons=config["target"]["horizons"],
            up_threshold=config["target"]["up_threshold"],
            down_threshold=config["target"]["down_threshold"],
        )
        
        if config["target"]["balance_classes"]:
            df = feature_engine.balance_classes(df, target_col="target_5m", method=config["target"]["balancing_method"])
        
        df.to_parquet(processed_path)
    
    print(f"\n[2/5] Loading features from {processed_path}")
    df = pd.read_parquet(processed_path)
    
    # Get feature columns
    exclude_cols = {"open", "high", "low", "close", "volume", "quote_volume", "trades", 
                   "taker_buy_base", "taker_buy_quote", "target_5m", "target_10m", "target_15m",
                   "forward_return_5m", "forward_return_10m", "forward_return_15m",
                   "hour", "minute", "day_of_week", "mid_price", "sample_weight", "regime", "rolling_vol"}
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Split data
    print("\n[3/5] Preparing datasets...")
    train_df, val_df, test_df = create_train_val_test_split(df, 0.70, 0.15, 0.15)
    
    # Apply profit optimizations
    trainer = ProfitOptimizedTrainer(config)
    
    # Filter by regime if enabled
    if config.get('profit_rules', {}).get('trade_only_high_vol'):
        print("  Filtering for high-volatility regime...")
        test_df = trainer.filter_by_regime(test_df, 'high')
    
    # Filter by trading hours (only if list is not empty)
    trade_hours = config.get('profit_rules', {}).get('trade_hours_utc', [])
    if trade_hours and len(trade_hours) > 0:
        print(f"  Filtering for optimal trading hours: {trade_hours}...")
        test_df = trainer.filter_by_trading_hours(test_df, trade_hours)
    
    # Create datasets
    train_dataset = TimeSeriesDataset(train_df, feature_cols, "target_5m", config["sequence"]["lookback_minutes"])
    val_dataset = TimeSeriesDataset(val_df, feature_cols, "target_5m", config["sequence"]["lookback_minutes"], train_dataset.scaler)
    test_dataset = TimeSeriesDataset(test_df, feature_cols, "target_5m", config["sequence"]["lookback_minutes"], train_dataset.scaler)
    
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    
    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    
    # Feature selection (only if enabled and not False)
    use_top_features = config.get('profit_rules', {}).get('use_top_n_features')
    if use_top_features and use_top_features != False:
        top_n = use_top_features if isinstance(use_top_features, int) else 30
        X_sample, y_sample = train_dataset.feature_values[:1000], train_dataset.targets[:1000]
        top_features = trainer.select_top_features(None, feature_cols, X_sample, y_sample, top_n=top_n)
        # Recreate datasets with top features only
        train_dataset = TimeSeriesDataset(train_df, top_features, "target_5m", config["sequence"]["lookback_minutes"])
        val_dataset = TimeSeriesDataset(val_df, top_features, "target_5m", config["sequence"]["lookback_minutes"], train_dataset.scaler)
        test_dataset = TimeSeriesDataset(test_df, top_features, "target_5m", config["sequence"]["lookback_minutes"], train_dataset.scaler)
        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
        feature_cols = top_features
    
    # Train model
    print(f"\n[4/5] Training {args.model}...")
    model = BaselineModelV2(args.model, config)
    model.train(train_loader, val_loader)
    
    # Evaluate with profit optimizations
    print("\n[5/5] Profit-optimized evaluation...")
    
    X_test = test_dataset.feature_values
    y_test = test_dataset.targets
    y_pred, y_pred_proba = model.predict(X_test)
    
    test_indices = test_df.index[config["sequence"]["lookback_minutes"]:][:len(y_pred)]
    actual_returns = test_df.loc[test_indices, "forward_return_5m"].values
    
    valid_mask = ~np.isnan(actual_returns)
    y_pred = y_pred[valid_mask]
    y_pred_proba = y_pred_proba[valid_mask]
    actual_returns = actual_returns[valid_mask]
    
    # Apply risk management
    adjusted_returns = trainer.apply_risk_management(y_pred, y_pred_proba, actual_returns)
    
    # Trading simulation
    simulator = TradingSimulator(
        initial_capital=10000,
        trading_cost=config["evaluation"]["trading_cost"],
        min_confidence=config["training"]["min_confidence"],
    )
    
    results = simulator.simulate(y_pred, y_pred_proba, adjusted_returns)
    
    print("\n" + "=" * 80)
    print("💰 PROFIT-OPTIMIZED RESULTS")
    print("=" * 80)
    simulator.print_results(results)
    
    print("\n📊 Performance Analysis:")
    print(f"  Confidence threshold: {config['training']['min_confidence']:.0%}")
    print(f"  Trades executed: {results['n_trades']} / {len(y_pred)} ({results['n_trades']/len(y_pred)*100:.1f}%)")
    print(f"  Win rate: {results['win_rate']:.1%}")
    print(f"  Total return: {results['total_return_pct']:+.2f}%")
    
    if results['total_return_pct'] > 0:
        print("\n✅ PROFITABLE! Strategy is making money.")
    else:
        print("\n⚠️  Still losing. Recommendations:")
        print("    1. Fetch more data (60-90 days)")
        print("    2. Increase confidence to 0.75 or 0.80")
        print("    3. Try different hours/regimes")
    
    print("=" * 80)


if __name__ == "__main__":
    main()

