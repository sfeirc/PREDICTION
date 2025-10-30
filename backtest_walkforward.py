"""
🔄 WALK-FORWARD BACKTESTING ENGINE

Most realistic backtesting method:
- Trains on historical data
- Tests on future data
- Rolls forward continuously
- Avoids look-ahead bias

Expected: Realistic performance estimates (usually 20-30% lower than naive backtest)
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class WalkForwardBacktest:
    """
    Walk-forward backtesting engine.
    
    Most realistic way to backtest:
    1. Train on window 1
    2. Test on window 2
    3. Roll forward
    4. Repeat
    """
    
    def __init__(
        self,
        train_days: int = 90,
        test_days: int = 7,
        step_days: int = 7,
        anchored: bool = False
    ):
        """
        Args:
            train_days: Days to train on
            test_days: Days to test on
            step_days: Days to step forward
            anchored: If True, use expanding window (train from start)
                     If False, use rolling window (fixed train size)
        """
        self.train_days = train_days
        self.test_days = test_days
        self.step_days = step_days
        self.anchored = anchored
        
        self.results = []
        
        logger.info(f"🔄 Walk-Forward Backtest initialized")
        logger.info(f"   Train: {train_days} days, Test: {test_days} days")
        logger.info(f"   Step: {step_days} days, Anchored: {anchored}")
    
    def run(
        self,
        df: pd.DataFrame,
        train_func: callable,
        predict_func: callable,
        evaluate_func: callable
    ) -> Dict:
        """
        Run walk-forward backtest.
        
        Args:
            df: DataFrame with datetime index and features/target
            train_func: Function(train_data) -> model
            predict_func: Function(model, test_data) -> predictions
            evaluate_func: Function(predictions, actuals) -> metrics
        
        Returns:
            Dict with overall results and per-window results
        """
        logger.info(f"🚀 Starting walk-forward backtest...")
        logger.info(f"   Data: {len(df)} samples, {df.index[0]} to {df.index[-1]}")
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")
        
        # Calculate windows
        start_date = df.index[0]
        end_date = df.index[-1]
        
        current_date = start_date + timedelta(days=self.train_days)
        window_num = 0
        
        while current_date + timedelta(days=self.test_days) <= end_date:
            window_num += 1
            
            # Define train window
            if self.anchored:
                # Expanding window: train from start
                train_start = start_date
            else:
                # Rolling window: fixed train size
                train_start = current_date - timedelta(days=self.train_days)
            
            train_end = current_date
            
            # Define test window
            test_start = current_date
            test_end = current_date + timedelta(days=self.test_days)
            
            # Extract data
            train_data = df[train_start:train_end]
            test_data = df[test_start:test_end]
            
            if len(train_data) < 100 or len(test_data) < 10:
                logger.warning(f"   Window {window_num}: Insufficient data, skipping")
                current_date += timedelta(days=self.step_days)
                continue
            
            logger.info(
                f"\n📊 Window {window_num}:"
            )
            logger.info(
                f"   Train: {train_start.date()} to {train_end.date()} "
                f"({len(train_data)} samples)"
            )
            logger.info(
                f"   Test:  {test_start.date()} to {test_end.date()} "
                f"({len(test_data)} samples)"
            )
            
            try:
                # Train model
                model = train_func(train_data)
                
                # Make predictions
                predictions = predict_func(model, test_data)
                
                # Evaluate
                metrics = evaluate_func(predictions, test_data)
                
                # Store results
                self.results.append({
                    'window': window_num,
                    'train_start': train_start,
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'train_size': len(train_data),
                    'test_size': len(test_data),
                    'metrics': metrics,
                    'predictions': predictions
                })
                
                logger.info(f"   ✅ Metrics: {metrics}")
                
            except Exception as e:
                logger.error(f"   ❌ Error in window {window_num}: {e}")
            
            # Move to next window
            current_date += timedelta(days=self.step_days)
        
        # Aggregate results
        overall_results = self._aggregate_results()
        
        logger.info(f"\n✅ Walk-forward backtest complete!")
        logger.info(f"   Total windows: {len(self.results)}")
        logger.info(f"   Overall metrics: {overall_results}")
        
        return {
            'overall': overall_results,
            'windows': self.results
        }
    
    def _aggregate_results(self) -> Dict:
        """Aggregate metrics across all windows"""
        if not self.results:
            return {}
        
        # Extract metrics
        all_metrics = {}
        for result in self.results:
            for key, value in result['metrics'].items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)
        
        # Calculate statistics
        aggregated = {}
        for key, values in all_metrics.items():
            aggregated[f'{key}_mean'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)
            aggregated[f'{key}_min'] = np.min(values)
            aggregated[f'{key}_max'] = np.max(values)
        
        return aggregated
    
    def plot_results(self, metric: str = 'accuracy', save_path: str = None):
        """
        Plot walk-forward results over time.
        """
        if not self.results:
            logger.warning("No results to plot")
            return
        
        # Extract data
        windows = [r['window'] for r in self.results]
        test_dates = [r['test_start'] for r in self.results]
        values = [r['metrics'].get(metric, 0) for r in self.results]
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: Metric over time
        ax1.plot(test_dates, values, 'bo-', alpha=0.6, label=f'{metric}')
        ax1.axhline(np.mean(values), color='r', linestyle='--', label='Mean')
        ax1.fill_between(
            test_dates,
            np.mean(values) - np.std(values),
            np.mean(values) + np.std(values),
            alpha=0.2, color='r', label='±1 std'
        )
        ax1.set_xlabel('Test Period Start')
        ax1.set_ylabel(metric.capitalize())
        ax1.set_title(f'Walk-Forward {metric.capitalize()} Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Distribution
        ax2.hist(values, bins=20, alpha=0.7, edgecolor='black')
        ax2.axvline(np.mean(values), color='r', linestyle='--', label=f'Mean: {np.mean(values):.3f}')
        ax2.set_xlabel(metric.capitalize())
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Distribution of {metric.capitalize()} Across Windows')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = f'logs/walkforward_{metric}.png'
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"📊 Plot saved to {save_path}")
        plt.close()
    
    def generate_report(self, save_path: str = 'logs/walkforward_report.txt'):
        """Generate comprehensive backtest report"""
        if not self.results:
            logger.warning("No results to report")
            return
        
        report = []
        report.append("="*80)
        report.append("🔄 WALK-FORWARD BACKTEST REPORT")
        report.append("="*80)
        report.append("")
        
        # Configuration
        report.append("📋 Configuration:")
        report.append(f"   Train Days: {self.train_days}")
        report.append(f"   Test Days: {self.test_days}")
        report.append(f"   Step Days: {self.step_days}")
        report.append(f"   Anchored: {self.anchored}")
        report.append(f"   Total Windows: {len(self.results)}")
        report.append("")
        
        # Overall metrics
        overall = self._aggregate_results()
        report.append("📊 Overall Metrics:")
        for key, value in sorted(overall.items()):
            if '_mean' in key:
                metric = key.replace('_mean', '')
                mean = value
                std = overall.get(f'{metric}_std', 0)
                report.append(f"   {metric}: {mean:.4f} ± {std:.4f}")
        report.append("")
        
        # Per-window results
        report.append("📈 Per-Window Results:")
        report.append("")
        
        for result in self.results:
            report.append(f"Window {result['window']}:")
            report.append(f"   Test Period: {result['test_start'].date()} to {result['test_end'].date()}")
            report.append(f"   Samples: {result['test_size']}")
            report.append(f"   Metrics:")
            for key, value in sorted(result['metrics'].items()):
                report.append(f"      {key}: {value:.4f}")
            report.append("")
        
        report.append("="*80)
        
        # Save report
        report_text = "\n".join(report)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"📄 Report saved to {save_path}")
        
        return report_text


def test_walkforward():
    """Test walk-forward backtesting"""
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    
    logging.basicConfig(level=logging.INFO)
    
    print("="*80)
    print("🔄 TESTING WALK-FORWARD BACKTEST")
    print("="*80)
    
    # Generate dummy data
    dates = pd.date_range('2023-01-01', periods=200, freq='D')
    n_features = 20
    
    df = pd.DataFrame(
        np.random.randn(200, n_features),
        index=dates,
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    df['target'] = np.random.randint(0, 2, 200)
    
    # Define functions
    def train_func(train_data):
        X = train_data[[c for c in train_data.columns if c != 'target']]
        y = train_data['target']
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        return model
    
    def predict_func(model, test_data):
        X = test_data[[c for c in test_data.columns if c != 'target']]
        return model.predict_proba(X)[:, 1]
    
    def evaluate_func(predictions, test_data):
        y_true = test_data['target'].values
        y_pred = (predictions > 0.5).astype(int)
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'auc': roc_auc_score(y_true, predictions)
        }
    
    # Run backtest
    wf = WalkForwardBacktest(
        train_days=90,
        test_days=7,
        step_days=7,
        anchored=False
    )
    
    results = wf.run(df, train_func, predict_func, evaluate_func)
    
    print(f"\n📊 Results:")
    print(f"   Windows tested: {len(results['windows'])}")
    print(f"   Overall metrics:")
    for key, value in results['overall'].items():
        if '_mean' in key:
            print(f"      {key}: {value:.4f}")
    
    # Generate plots and report
    wf.plot_results('accuracy')
    wf.plot_results('auc')
    wf.generate_report()
    
    print("\n✅ Walk-forward backtest test complete!")
    print("="*80)


if __name__ == "__main__":
    test_walkforward()

