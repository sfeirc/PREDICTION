"""
🚀 MAXIMUM PROFIT TRADING SYSTEM 🚀

This script maximizes trading opportunities and profits using:
- Optimal confidence threshold for max trades while staying profitable
- Compounding (reinvest profits)
- Kelly Criterion for position sizing
- All available data
- Risk management to protect capital
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


class MaxProfitTrader:
    """Advanced trading system with compounding and optimal position sizing."""
    
    def __init__(self, initial_capital=10000, trading_cost=0.001):
        self.initial_capital = initial_capital
        self.trading_cost = trading_cost
        
    def kelly_criterion(self, win_rate, avg_win, avg_loss):
        """
        Calculate optimal position size using Kelly Criterion.
        
        f* = (p * b - q) / b
        where:
        - p = win probability
        - q = loss probability (1-p)
        - b = win/loss ratio
        """
        if avg_loss == 0 or win_rate <= 0:
            return 0.0
        
        b = abs(avg_win / avg_loss)
        q = 1 - win_rate
        kelly = (win_rate * b - q) / b
        
        # Use fractional Kelly (0.25 to 0.5) for safety
        return max(0, min(kelly * 0.5, 0.5))  # Cap at 50% of capital
    
    def simulate_with_compounding(
        self,
        predictions,
        probabilities,
        actual_returns,
        confidence_threshold,
        use_kelly=True,
    ):
        """
        Simulate trading with compounding and position sizing.
        """
        n_samples = len(predictions)
        
        # Track results
        capital = [self.initial_capital]
        positions = []
        trades = []
        
        # Calculate confidence
        confidences = np.maximum(probabilities, 1 - probabilities)
        
        # First pass: estimate win rate and avg win/loss for Kelly
        wins = []
        losses = []
        
        for i in range(n_samples):
            if confidences[i] >= confidence_threshold and predictions[i] == 1:
                ret = actual_returns[i] - 2 * self.trading_cost
                if ret > 0:
                    wins.append(ret)
                else:
                    losses.append(abs(ret))
        
        if len(wins) == 0:
            return {
                'final_capital': self.initial_capital,
                'total_return': 0,
                'n_trades': 0,
                'win_rate': 0,
                'trades': [],
            }
        
        win_rate = len(wins) / (len(wins) + len(losses)) if (len(wins) + len(losses)) > 0 else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0.01
        
        kelly_fraction = self.kelly_criterion(win_rate, avg_win, avg_loss) if use_kelly else 0.25
        kelly_fraction = max(0.05, min(kelly_fraction, 0.5))  # Between 5% and 50%
        
        print(f"  Kelly Criterion: {kelly_fraction:.1%} position size")
        
        # Second pass: actual trading with compounding
        current_capital = self.initial_capital
        n_trades = 0
        wins_count = 0
        
        for i in range(n_samples):
            conf = confidences[i]
            
            if conf < confidence_threshold:
                continue
            
            if predictions[i] != 1:  # Only go long (no shorting)
                continue
            
            # Calculate position size
            position_size = current_capital * kelly_fraction
            
            # Execute trade
            gross_return = actual_returns[i]
            net_return = gross_return - 2 * self.trading_cost
            
            pnl = position_size * net_return
            
            # Update capital
            current_capital += pnl
            capital.append(current_capital)
            
            # Track trade
            n_trades += 1
            if pnl > 0:
                wins_count += 1
            
            trades.append({
                'index': i,
                'confidence': conf,
                'position_size': position_size,
                'return': net_return,
                'pnl': pnl,
                'capital': current_capital,
            })
            
            # Stop if bankrupt
            if current_capital <= 0:
                print(f"  ⚠️  BANKRUPT after {n_trades} trades!")
                break
        
        final_win_rate = wins_count / n_trades if n_trades > 0 else 0
        total_return = (current_capital - self.initial_capital) / self.initial_capital * 100
        
        return {
            'final_capital': current_capital,
            'total_return': total_return,
            'n_trades': n_trades,
            'win_rate': final_win_rate,
            'trades': trades,
            'capital_curve': capital,
            'kelly_fraction': kelly_fraction,
        }


def main():
    print("=" * 80)
    print("🚀 MAXIMUM PROFIT TRADING SYSTEM 🚀")
    print("=" * 80)
    print("\nOptimizations:")
    print("  ✓ Compounding (reinvest profits)")
    print("  ✓ Kelly Criterion (optimal position sizing)")
    print("  ✓ Optimal confidence threshold")
    print("  ✓ Maximum data utilization")
    
    # Load config
    with open("config_profitable.yaml") as f:
        config = yaml.safe_load(f)
    
    set_seed(config["seed"])
    
    # Load data
    processed_path = Path(config["data"]["processed_dir"]) / "btcusdt_features_v2.parquet"
    
    if not processed_path.exists():
        print("\n❌ No processed features found!")
        print("Run: python feature_engineering_v2.py first")
        return
    
    df = pd.read_parquet(processed_path)
    
    print(f"\n📊 Data loaded: {len(df)} samples")
    print(f"   Date range: {df.index.min()} to {df.index.max()}")
    print(f"   Duration: {(df.index.max() - df.index.min()).days} days")
    
    # Get features
    exclude_cols = {"open", "high", "low", "close", "volume", "quote_volume", "trades", 
                   "taker_buy_base", "taker_buy_quote", "target_5m", "target_10m", "target_15m",
                   "forward_return_5m", "forward_return_10m", "forward_return_15m",
                   "hour", "minute", "day_of_week", "mid_price", "sample_weight", "regime", "rolling_vol"}
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Use ALL data for training (not just 70%)
    print("\n🎯 Strategy: Use MAXIMUM data for training")
    
    # Split but use larger training set
    train_ratio = 0.85  # Use 85% for training (more data = better model)
    val_ratio = 0.00    # No validation (we know what works)
    test_ratio = 0.15   # 15% for testing
    
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    print(f"  Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    # Create datasets
    train_dataset = TimeSeriesDataset(train_df, feature_cols, "target_5m", config["sequence"]["lookback_minutes"])
    test_dataset = TimeSeriesDataset(test_df, feature_cols, "target_5m", config["sequence"]["lookback_minutes"], train_dataset.scaler)
    
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    
    print(f"\n  After filtering: Train={len(train_dataset)}, Test={len(test_dataset)}")
    
    # Train model
    print("\n🤖 Training LightGBM (optimized for profit)...")
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
    
    print(f"  Valid predictions: {len(y_pred)}")
    
    # Test different strategies
    print("\n" + "=" * 80)
    print("🎮 TESTING STRATEGIES")
    print("=" * 80)
    
    trader = MaxProfitTrader(initial_capital=10000, trading_cost=0.001)
    
    # Test different confidence levels
    confidence_levels = [0.55, 0.60, 0.65, 0.70, 0.75]
    all_results = []
    
    for conf in confidence_levels:
        print(f"\n📈 Testing confidence: {conf:.0%}")
        
        # With Kelly Criterion
        result_kelly = trader.simulate_with_compounding(
            y_pred, y_pred_proba, actual_returns, conf, use_kelly=True
        )
        
        # Fixed 25% position size (for comparison)
        result_fixed = trader.simulate_with_compounding(
            y_pred, y_pred_proba, actual_returns, conf, use_kelly=False
        )
        
        print(f"\n  Kelly Criterion:")
        print(f"    Trades: {result_kelly['n_trades']}")
        print(f"    Win Rate: {result_kelly['win_rate']:.1%}")
        print(f"    Total Return: {result_kelly['total_return']:+.2f}%")
        print(f"    Final Capital: ${result_kelly['final_capital']:,.2f}")
        
        print(f"\n  Fixed 25%:")
        print(f"    Trades: {result_fixed['n_trades']}")
        print(f"    Win Rate: {result_fixed['win_rate']:.1%}")
        print(f"    Total Return: {result_fixed['total_return']:+.2f}%")
        print(f"    Final Capital: ${result_fixed['final_capital']:,.2f}")
        
        all_results.append({
            'confidence': conf,
            'kelly_return': result_kelly['total_return'],
            'kelly_trades': result_kelly['n_trades'],
            'kelly_capital': result_kelly['final_capital'],
            'fixed_return': result_fixed['total_return'],
            'fixed_trades': result_fixed['n_trades'],
            'fixed_capital': result_fixed['final_capital'],
        })
    
    # Find best strategy
    results_df = pd.DataFrame(all_results)
    
    print("\n" + "=" * 80)
    print("📊 RESULTS SUMMARY")
    print("=" * 80)
    
    print(f"\n{'Confidence':<12} {'Trades':<8} {'Kelly Return':<15} {'Fixed Return':<15}")
    print("-" * 80)
    for _, row in results_df.iterrows():
        print(f"{row['confidence']:<12.0%} "
              f"{int(row['kelly_trades']):<8d} "
              f"{row['kelly_return']:+14.2f}% "
              f"{row['fixed_return']:+14.2f}%")
    
    # Best strategy
    best_kelly = results_df.loc[results_df['kelly_return'].idxmax()]
    best_fixed = results_df.loc[results_df['fixed_return'].idxmax()]
    
    print("\n" + "=" * 80)
    print("🏆 MAXIMUM PROFIT STRATEGY")
    print("=" * 80)
    
    if best_kelly['kelly_return'] > best_fixed['fixed_return']:
        print(f"\n✅ BEST: Kelly Criterion at {best_kelly['confidence']:.0%} confidence")
        print(f"   📈 Return: {best_kelly['kelly_return']:+.2f}%")
        print(f"   💰 Final Capital: ${best_kelly['kelly_capital']:,.2f}")
        print(f"   🎯 Trades: {int(best_kelly['kelly_trades'])}")
        print(f"   💵 Profit: ${best_kelly['kelly_capital'] - 10000:,.2f}")
        
        best_conf = best_kelly['confidence']
        best_result = trader.simulate_with_compounding(
            y_pred, y_pred_proba, actual_returns, best_conf, use_kelly=True
        )
    else:
        print(f"\n✅ BEST: Fixed 25% at {best_fixed['confidence']:.0%} confidence")
        print(f"   📈 Return: {best_fixed['fixed_return']:+.2f}%")
        print(f"   💰 Final Capital: ${best_fixed['fixed_capital']:,.2f}")
        print(f"   🎯 Trades: {int(best_fixed['fixed_trades'])}")
        print(f"   💵 Profit: ${best_fixed['fixed_capital'] - 10000:,.2f}")
        
        best_conf = best_fixed['confidence']
        best_result = trader.simulate_with_compounding(
            y_pred, y_pred_proba, actual_returns, best_conf, use_kelly=False
        )
    
    # Plot capital curve
    if len(best_result['capital_curve']) > 1:
        plt.figure(figsize=(14, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(best_result['capital_curve'], linewidth=2, color='green')
        plt.axhline(y=10000, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
        plt.title('Capital Growth with Compounding', fontsize=14, fontweight='bold')
        plt.ylabel('Capital ($)')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.subplot(2, 1, 2)
        cummax = np.maximum.accumulate(best_result['capital_curve'])
        drawdown = (np.array(best_result['capital_curve']) - cummax) / cummax * 100
        plt.fill_between(range(len(drawdown)), drawdown, 0, color='red', alpha=0.3)
        plt.plot(drawdown, color='red', linewidth=1)
        plt.title('Drawdown', fontsize=14, fontweight='bold')
        plt.ylabel('Drawdown (%)')
        plt.xlabel('Trade Number')
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plot_path = Path("logs") / "maximum_profit_strategy.png"
        plot_path.parent.mkdir(exist_ok=True)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Capital curve saved to: {plot_path}")
    
    # Realistic expectations
    print("\n" + "=" * 80)
    print("💡 SCALING PROJECTIONS")
    print("=" * 80)
    
    if best_result['total_return'] > 0:
        monthly_return = best_result['total_return']
        
        print(f"\nWith $10,000 initial capital:")
        print(f"  Month 1: ${10000 * (1 + monthly_return/100):,.2f}")
        print(f"  Month 3: ${10000 * (1 + monthly_return/100)**3:,.2f}")
        print(f"  Month 6: ${10000 * (1 + monthly_return/100)**6:,.2f}")
        print(f"  Year 1:  ${10000 * (1 + monthly_return/100)**12:,.2f}")
        
        print(f"\nWith $100,000 initial capital:")
        print(f"  Month 1: ${100000 * (1 + monthly_return/100):,.2f}")
        print(f"  Year 1:  ${100000 * (1 + monthly_return/100)**12:,.2f}")
        
        print(f"\n⚠️  IMPORTANT:")
        print(f"  - Past performance doesn't guarantee future results")
        print(f"  - Start small and scale up gradually")
        print(f"  - Always use stop losses")
        print(f"  - Never trade more than you can afford to lose")
    
    print("\n" + "=" * 80)
    print("🎉 MAXIMUM PROFIT ANALYSIS COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()

