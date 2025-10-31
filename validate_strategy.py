"""
📊 COMPREHENSIVE STRATEGY VALIDATION

Run realistic walk-forward backtest with proper evaluation metrics.
This shows REAL performance before going live.
"""

import sys
import io
import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from data_manager_worldclass import DataManagerWorldClass
from feature_engine_worldclass import FeatureEngineWorldClass
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import TimeSeriesSplit


def load_best_params() -> dict:
    """Load optimized parameters"""
    best_path = Path('logs/best_settings.yaml')
    if not best_path.exists():
        logger.warning("No best_settings.yaml found, using defaults")
        return {
            'learning_rate': 0.03,
            'max_depth': 8,
            'num_leaves': 64,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
    
    with open(best_path, 'r', encoding='utf-8') as f:
        best = yaml.safe_load(f) or {}
    
    params = best.get('best_lightgbm_params', {})
    if not params:
        logger.warning("No best_lightgbm_params found, using defaults")
        return {
            'learning_rate': 0.03,
            'max_depth': 8,
            'num_leaves': 64,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
    
    return params


def calculate_trading_metrics(pred_proba, y_true, y_pred, prices):
    """Calculate realistic trading metrics"""
    # Convert probabilities to signals
    confidence_threshold = 0.68
    signals = []
    trades = []
    
    for i in range(len(pred_proba)):
        prob = pred_proba[i]
        if prob > confidence_threshold:
            signals.append(1)  # Buy
        elif prob < (1 - confidence_threshold):
            signals.append(-1)  # Sell
        else:
            signals.append(0)  # Hold
    
    # Simple PnL calculation (simplified)
    positions = []
    pnl = 0
    entry_price = None
    
    for i in range(1, len(signals)):
        if signals[i] == 1 and signals[i-1] != 1:  # Entry
            entry_price = prices.iloc[i]
            positions.append(('LONG', entry_price))
        elif signals[i] == -1 and len(positions) > 0:  # Exit
            if positions[-1][0] == 'LONG':
                exit_price = prices.iloc[i]
                trade_pnl = (exit_price - entry_price) / entry_price if entry_price else 0
                pnl += trade_pnl
                trades.append({
                    'entry': entry_price,
                    'exit': exit_price,
                    'pnl': trade_pnl,
                    'type': 'LONG'
                })
                positions.pop()
    
    # Close remaining positions
    if positions and len(prices) > 0:
        for pos_type, entry_price in positions:
            exit_price = prices.iloc[-1]
            trade_pnl = (exit_price - entry_price) / entry_price if entry_price else 0
            pnl += trade_pnl
    
    total_return = pnl
    win_trades = sum(1 for t in trades if t['pnl'] > 0)
    win_rate = win_trades / len(trades) if trades else 0
    
    # Sharpe proxy (simplified)
    returns = pd.Series([t['pnl'] for t in trades]) if trades else pd.Series([0])
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 1 and returns.std() > 0 else 0
    
    return {
        'total_return': total_return,
        'num_trades': len(trades),
        'win_rate': win_rate,
        'sharpe': sharpe,
        'trades': trades
    }


def run_walk_forward_validation(df, feature_cols, target_col, best_params, train_days=60, test_days=7, step_days=7):
    """Run walk-forward validation"""
    logger.info(f"\n{'='*80}")
    logger.info("🔄 WALK-FORWARD VALIDATION")
    logger.info(f"{'='*80}")
    logger.info(f"Train: {train_days} days, Test: {test_days} days, Step: {step_days} days")
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        else:
            logger.error("No datetime index or timestamp column found")
            return None
    
    df = df.sort_index()
    
    # Get date range
    start_date = df.index.min()
    end_date = df.index.max()
    total_days = (end_date - start_date).days
    
    logger.info(f"Data range: {start_date} to {end_date} ({total_days} days)")
    
    windows = []
    current_start = start_date
    
    while current_start + pd.Timedelta(days=train_days + test_days) <= end_date:
        train_end = current_start + pd.Timedelta(days=train_days)
        test_start = train_end
        test_end = min(test_start + pd.Timedelta(days=test_days), end_date)
        
        windows.append({
            'train_start': current_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end
        })
        
        current_start += pd.Timedelta(days=step_days)
    
    logger.info(f"Total windows: {len(windows)}")
    
    if len(windows) == 0:
        logger.warning("No valid windows found! Need more data.")
        logger.info(f"Required: {train_days + test_days} days, Available: {total_days} days")
        return None
    
    all_results = []
    
    for i, window in enumerate(windows):
        logger.info(f"\n--- Window {i+1}/{len(windows)} ---")
        logger.info(f"Train: {window['train_start'].date()} to {window['train_end'].date()}")
        logger.info(f"Test:  {window['test_start'].date()} to {window['test_end'].date()}")
        
        # Split data
        train_mask = (df.index >= window['train_start']) & (df.index < window['train_end'])
        test_mask = (df.index >= window['test_start']) & (df.index < window['test_end'])
        
        df_train = df[train_mask].copy()
        df_test = df[test_mask].copy()
        
        if len(df_train) < 100 or len(df_test) < 10:
            logger.warning(f"Skipping window {i+1}: insufficient data (train={len(df_train)}, test={len(df_test)})")
            continue
        
        # Prepare features
        X_train = df_train[feature_cols].values
        y_train = df_train[target_col].values
        X_test = df_test[feature_cols].values
        y_test = df_test[target_col].values
        
        # Train model
        model = LGBMClassifier(
            n_estimators=300,
            learning_rate=float(best_params.get('learning_rate', 0.03)),
            max_depth=int(best_params.get('max_depth', 8)),
            num_leaves=int(best_params.get('num_leaves', 64)),
            min_child_samples=int(best_params.get('min_child_samples', 20)),
            subsample=float(best_params.get('subsample', 0.8)),
            colsample_bytree=float(best_params.get('colsample_bytree', 0.8)),
            random_state=42,
            verbose=-1
        )
        
        model.fit(X_train, y_train)
        
        # Predict
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0
        f1 = f1_score(y_test, y_pred) if len(np.unique(y_test)) > 1 else 0
        
        # Trading metrics
        prices = df_test['close'] if 'close' in df_test.columns else pd.Series([1.0] * len(df_test))
        trading_metrics = calculate_trading_metrics(y_pred_proba, y_test, y_pred, prices)
        
        result = {
            'window': i+1,
            'train_start': window['train_start'],
            'train_end': window['train_end'],
            'test_start': window['test_start'],
            'test_end': window['test_end'],
            'accuracy': acc,
            'auc': auc,
            'f1': f1,
            'total_return': trading_metrics['total_return'],
            'num_trades': trading_metrics['num_trades'],
            'win_rate': trading_metrics['win_rate'],
            'sharpe': trading_metrics['sharpe']
        }
        
        all_results.append(result)
        
        logger.info(f"  Accuracy: {acc:.4f}, AUC: {auc:.4f}, Return: {trading_metrics['total_return']:+.2%}, Trades: {trading_metrics['num_trades']}")
    
    # Aggregate results
    if not all_results:
        logger.error("No valid windows completed!")
        return None
    
    results_df = pd.DataFrame(all_results)
    
    logger.info(f"\n{'='*80}")
    logger.info("📊 AGGREGATE RESULTS")
    logger.info(f"{'='*80}")
    
    for metric in ['accuracy', 'auc', 'f1', 'total_return', 'win_rate', 'sharpe']:
        mean_val = results_df[metric].mean()
        std_val = results_df[metric].std()
        logger.info(f"{metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")
    
    logger.info(f"\nTotal Trades: {results_df['num_trades'].sum()}")
    logger.info(f"Avg Trades/Window: {results_df['num_trades'].mean():.1f}")
    
    return {
        'windows': results_df,
        'overall': {
            'accuracy_mean': results_df['accuracy'].mean(),
            'auc_mean': results_df['auc'].mean(),
            'f1_mean': results_df['f1'].mean(),
            'return_mean': results_df['total_return'].mean(),
            'win_rate_mean': results_df['win_rate'].mean(),
            'sharpe_mean': results_df['sharpe'].mean(),
            'total_trades': int(results_df['num_trades'].sum())
        }
    }


def main():
    """Main validation pipeline"""
    logger.info("="*80)
    logger.info("📊 STRATEGY VALIDATION - COMPREHENSIVE BACKTEST")
    logger.info("="*80)
    
    # Load config
    with open('config_ultimate.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}
    
    # Ensure correlation_pairs exists
    config.setdefault('data', {})
    config['data'].setdefault('correlation_pairs', ['ETHUSDT', 'BNBUSDT'])
    
    # 1. Fetch data
    logger.info("\n📥 Fetching data...")
    dm = DataManagerWorldClass(config)
    data = dm.fetch_all_data()
    
    # 2. Create features
    logger.info("\n🔧 Creating features...")
    fe = FeatureEngineWorldClass(config)
    df = fe.create_features(data)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    logger.info(f"Dataset: {len(df)} samples, {len(df.columns)} columns")
    
    # 3. Identify features and target
    feature_cols = [c for c in df.columns if not c.startswith('target_')]
    target_cols = [c for c in df.columns if c.startswith('target_')]
    
    if not target_cols:
        logger.error("No target columns found!")
        return
    
    target_col = target_cols[0]
    logger.info(f"Features: {len(feature_cols)}, Target: {target_col}")
    
    # 4. Load best params
    best_params = load_best_params()
    logger.info(f"Using optimized params: {best_params}")
    
    # 5. Run walk-forward validation
    results = run_walk_forward_validation(
        df, feature_cols, target_col, best_params,
        train_days=20,  # Smaller windows to fit available data
        test_days=3,
        step_days=2
    )
    
    if results:
        # Save results
        results_path = Path('logs/validation_results.csv')
        results['windows'].to_csv(results_path, index=False)
        logger.info(f"\n✅ Results saved to {results_path}")
        
        # Summary
        overall = results['overall']
        logger.info("\n" + "="*80)
        logger.info("✅ VALIDATION COMPLETE")
        logger.info("="*80)
        logger.info(f"Average Accuracy: {overall['accuracy_mean']:.2%}")
        logger.info(f"Average AUC: {overall['auc_mean']:.2%}")
        logger.info(f"Average Return: {overall['return_mean']:+.2%}")
        logger.info(f"Average Win Rate: {overall['win_rate_mean']:.2%}")
        logger.info(f"Average Sharpe: {overall['sharpe_mean']:.2f}")
        logger.info(f"Total Trades: {overall['total_trades']}")
        
        # Decision
        logger.info("\n" + "="*80)
        if overall['auc_mean'] > 0.90 and overall['win_rate_mean'] > 0.55:
            logger.info("✅ STRATEGY VALIDATED - Ready for paper trading!")
        elif overall['auc_mean'] > 0.85:
            logger.info("⚠️  STRATEGY MARGINAL - Needs improvement before trading")
        else:
            logger.info("❌ STRATEGY WEAK - Requires significant enhancement")
        logger.info("="*80)
    else:
        logger.error("❌ Validation failed - check data and parameters")


if __name__ == '__main__':
    main()

