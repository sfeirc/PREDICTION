"""
Simple trading simulator to evaluate model predictions in a trading context.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import matplotlib.pyplot as plt


class TradingSimulator:
    """
    Simulate trading based on model predictions.
    
    Strategy:
    - Go long when model predicts "up" with high confidence
    - Go short (or flat) when model predicts "down" with high confidence
    - Stay flat when confidence is low
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        position_size: float = 1.0,  # Fraction of capital to use per trade
        trading_cost: float = 0.001,  # 0.1% per trade (taker fee)
        min_confidence: float = 0.6,  # Minimum confidence to trade
    ):
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.trading_cost = trading_cost
        self.min_confidence = min_confidence
    
    def simulate(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        actual_returns: np.ndarray,
        timestamps: np.ndarray = None,
    ) -> Dict:
        """
        Run trading simulation.
        
        Args:
            predictions: Binary predictions (0/1)
            probabilities: Prediction probabilities
            actual_returns: Actual forward returns
            timestamps: Optional timestamps
        
        Returns:
            Dictionary with simulation results
        """
        n_samples = len(predictions)
        
        # Initialize tracking arrays
        capital = np.zeros(n_samples + 1)
        capital[0] = self.initial_capital
        
        positions = np.zeros(n_samples)  # 1 = long, -1 = short, 0 = flat
        trade_pnl = np.zeros(n_samples)
        
        n_trades = 0
        n_long = 0
        n_short = 0
        
        for i in range(n_samples):
            # Check confidence
            confidence = max(probabilities[i], 1 - probabilities[i])
            
            if confidence < self.min_confidence:
                # Low confidence - stay flat
                positions[i] = 0
                capital[i + 1] = capital[i]
                continue
            
            # High confidence - take position
            if predictions[i] == 1:
                # Predict up - go long
                positions[i] = 1
                n_long += 1
            else:
                # Predict down - go short (or stay flat for crypto)
                # In crypto spot, we can't short, so stay flat
                positions[i] = 0
                capital[i + 1] = capital[i]
                continue
            
            # Execute trade
            n_trades += 1
            
            # Compute P&L
            gross_pnl = positions[i] * actual_returns[i]
            
            # Subtract trading costs (entry + exit)
            net_pnl = gross_pnl - 2 * self.trading_cost
            
            # Update capital
            trade_pnl[i] = net_pnl * capital[i] * self.position_size
            capital[i + 1] = capital[i] + trade_pnl[i]
        
        # Compute metrics
        total_return = (capital[-1] - self.initial_capital) / self.initial_capital
        
        # Sharpe ratio (annualized)
        if n_trades > 0:
            returns = np.diff(capital) / capital[:-1]
            returns = returns[returns != 0]  # Remove zero returns
            
            if len(returns) > 1:
                sharpe = np.mean(returns) / (np.std(returns) + 1e-8)
                sharpe_annualized = sharpe * np.sqrt(252 * 24 * 60)  # Assuming 1-min data
            else:
                sharpe_annualized = 0.0
        else:
            sharpe_annualized = 0.0
        
        # Max drawdown
        cummax = np.maximum.accumulate(capital)
        drawdown = (capital - cummax) / cummax
        max_drawdown = abs(drawdown.min())
        
        # Win rate
        winning_trades = (trade_pnl > 0).sum()
        win_rate = winning_trades / n_trades if n_trades > 0 else 0.0
        
        # Create results dictionary
        results = {
            "initial_capital": self.initial_capital,
            "final_capital": capital[-1],
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "n_trades": n_trades,
            "n_long": n_long,
            "n_short": n_short,
            "win_rate": win_rate,
            "sharpe_ratio": sharpe_annualized,
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": max_drawdown * 100,
            "avg_trade_pnl": trade_pnl[trade_pnl != 0].mean() if n_trades > 0 else 0.0,
            "capital_curve": capital,
            "positions": positions,
            "trade_pnl": trade_pnl,
        }
        
        return results
    
    def print_results(self, results: Dict):
        """Print simulation results."""
        print("\n" + "=" * 60)
        print("Trading Simulation Results")
        print("=" * 60)
        print(f"Initial Capital:    ${results['initial_capital']:,.2f}")
        print(f"Final Capital:      ${results['final_capital']:,.2f}")
        print(f"Total Return:       {results['total_return_pct']:+.2f}%")
        print(f"")
        print(f"Number of Trades:   {results['n_trades']}")
        print(f"  Long:             {results['n_long']}")
        print(f"  Short:            {results['n_short']}")
        print(f"Win Rate:           {results['win_rate']:.2%}")
        print(f"Avg Trade P&L:      ${results['avg_trade_pnl']:,.2f}")
        print(f"")
        print(f"Sharpe Ratio:       {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown:       {results['max_drawdown_pct']:.2f}%")
        print("=" * 60)
    
    def plot_results(self, results: Dict, save_path: str = None):
        """Plot capital curve and drawdown."""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        capital = results["capital_curve"]
        
        # Capital curve
        axes[0].plot(capital, linewidth=2)
        axes[0].axhline(y=self.initial_capital, color='gray', linestyle='--', label='Initial Capital')
        axes[0].set_title('Capital Curve', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Trade Number')
        axes[0].set_ylabel('Capital ($)')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Drawdown
        cummax = np.maximum.accumulate(capital)
        drawdown = (capital - cummax) / cummax * 100
        
        axes[1].fill_between(range(len(drawdown)), drawdown, 0, color='red', alpha=0.3)
        axes[1].plot(drawdown, color='red', linewidth=1)
        axes[1].set_title('Drawdown', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Trade Number')
        axes[1].set_ylabel('Drawdown (%)')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved trading simulation plot to {save_path}")
        else:
            plt.show()
        
        plt.close()


def backtest_predictions(
    predictions: np.ndarray,
    probabilities: np.ndarray,
    actual_returns: np.ndarray,
    config: Dict,
    timestamps: np.ndarray = None,
) -> Dict:
    """
    Run backtest on predictions.
    
    Args:
        predictions: Binary predictions
        probabilities: Prediction probabilities
        actual_returns: Actual forward returns
        config: Configuration dictionary
        timestamps: Optional timestamps
    
    Returns:
        Backtest results dictionary
    """
    simulator = TradingSimulator(
        initial_capital=10000.0,
        position_size=1.0,
        trading_cost=config["evaluation"].get("trading_cost", 0.001),
        min_confidence=config["training"].get("min_confidence", 0.6),
    )
    
    results = simulator.simulate(predictions, probabilities, actual_returns, timestamps)
    
    return results

