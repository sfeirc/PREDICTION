"""
AUTO OPTIMIZER

End-to-end pipeline:
1) Load or create features
2) Train base ensemble
3) Bayesian optimize LightGBM
4) Walk-forward backtest best params
5) Export best settings to logs/best_settings.yaml
6) Print KPI summary (aiming 40–45% monthly)
"""

import os
import sys
import io
import json
import yaml
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import project pieces lazily to avoid heavy import cost if not present
from bayesian_optimization import optimize_trading_bot
from backtest_walkforward import WalkForwardBacktest
from model_ensemble_worldclass import ModelEnsembleWorldClass
from feature_engine_worldclass import FeatureEngineWorldClass
from data_manager_worldclass import DataManagerWorldClass


DEFAULT_CONFIG = {
    'data': {
        'primary_pair': 'BTCUSDT',
        'portfolio_pairs': ['ETHUSDT', 'BNBUSDT', 'SOLUSDT'],
        'correlation_pairs': ['ETHUSDT', 'BNBUSDT'],
        'timeframes': {
            'primary': '1m',
            'analysis': ['1m', '5m', '15m', '1h', '4h', '1d']
        },
        'history_days': 90
    },
    'target': {
        'multi_horizon': True,
        'horizons': [5, 10, 15],
        'up_threshold': 0.002,
        'down_threshold': -0.002,
        'balance_classes': True,
        'balancing_method': 'downsample'
    },
    'models': {
        'ensemble': {
            'weights': {
                'lightgbm': 0.6,
                'xgboost': 0.4,
                'catboost': 0.0
            }
        }
    }
}


def ensure_dirs():
    Path('logs').mkdir(exist_ok=True)
    Path('models').mkdir(exist_ok=True)


def load_config(path: str = 'config_ultimate.yaml') -> dict:
    cfg = {}
    if Path(path).exists():
        with open(path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
    # Merge safe defaults
    for top_key in DEFAULT_CONFIG:
        section = cfg.get(top_key, {})
        defaults = DEFAULT_CONFIG[top_key]
        if isinstance(defaults, dict):
            for k, v in defaults.items():
                if isinstance(v, dict):
                    sub = section.get(k, {})
                    for sk, sv in v.items():
                        if sk not in sub:
                            sub[sk] = sv
                    section[k] = sub
                else:
                    if k not in section:
                        section[k] = v
            cfg[top_key] = section
        else:
            if top_key not in cfg:
                cfg[top_key] = defaults
    return cfg


def build_dataset(config: dict):
    logger.info('Fetching data and creating features...')
    dm = DataManagerWorldClass(config)
    data = dm.fetch_all_data()  # uses config timeframes
    fe = FeatureEngineWorldClass(config)
    df = fe.create_features(data)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    # Split features/target
    target_cols = [c for c in df.columns if c.startswith('target_')]
    if not target_cols:
        raise RuntimeError('No target_* columns found in features dataframe')
    target = target_cols[0]
    X = df.drop(columns=target_cols)
    y = df[target].astype(int)
    return df, X.values, y.values, X.columns.tolist(), target


def time_split(X, y, train_ratio=0.7, val_ratio=0.15):
    n = len(X)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def train_base_ensemble(config, df_features: pd.DataFrame):
    logger.info('Training base ensemble (LightGBM/XGBoost/CatBoost if available)...')
    # Ensure primary_horizon exists
    tgt = config.get('target', {})
    if 'primary_horizon' not in tgt:
        tgt['primary_horizon'] = 5
    config['target'] = tgt
    ensemble = ModelEnsembleWorldClass(config)
    _metrics = ensemble.train(df_features, config)
    return ensemble, _metrics


def bayes_optimize_lgbm(X_train, y_train, X_val, y_val):
    logger.info('Starting Bayesian optimization for LightGBM...')
    result = optimize_trading_bot(X_train, y_train, X_val, y_val, model_type='lightgbm')
    logger.info(f"Best BO params: {result['best_params']}")
    return result['best_params'], result['best_score']


def run_walk_forward(df: pd.DataFrame, best_params: dict):
    logger.info('Running walk-forward backtest on best params...')
    from sklearn.metrics import accuracy_score, roc_auc_score
    from lightgbm import LGBMClassifier

    features = [c for c in df.columns if not c.startswith('target_')]
    target = [c for c in df.columns if c.startswith('target_')][0]

    def train_func(train_df):
        X = train_df[features].values
        y = train_df[target].values
        model = LGBMClassifier(
            n_estimators=300,
            learning_rate=best_params.get('learning_rate', 0.01),
            max_depth=int(best_params.get('max_depth', 8)),
            num_leaves=int(best_params.get('num_leaves', 64)),
            min_child_samples=int(best_params.get('min_child_samples', 20)),
            subsample=best_params.get('subsample', 0.8),
            colsample_bytree=best_params.get('colsample_bytree', 0.8),
            random_state=42,
            verbose=-1
        )
        model.fit(X, y)
        return model

    def predict_func(model, test_df):
        X = test_df[features].values
        return model.predict_proba(X)[:, 1]

    def evaluate_func(pred_proba, test_df):
        y_true = test_df[target].values
        y_pred = (np.maximum(pred_proba, 1 - pred_proba) > 0.65).astype(int)
        # Accuracy & AUC
        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, pred_proba)
        # Proxy monthly return and sharpe (rough, scaled from win/loss)
        win_rate = float((y_pred == y_true).mean())
        # Very rough proxies for quick gating
        sharpe_proxy = (win_rate - 0.5) / 0.1 if win_rate > 0 else 0.0
        return {
            'accuracy': acc,
            'auc': auc,
            'win_rate': win_rate,
            'sharpe_proxy': sharpe_proxy
        }

    wf = WalkForwardBacktest(train_days=90, test_days=7, step_days=7, anchored=False)
    results = wf.run(df, train_func, predict_func, evaluate_func)
    wf.plot_results('accuracy')
    wf.plot_results('auc')
    wf.generate_report()
    return results


def export_best_settings(best_params: dict, metrics: dict, path='logs/best_settings.yaml'):
    payload = {
        'timestamp': int(time.time()),
        'best_lightgbm_params': best_params,
        'walkforward_overall': metrics.get('overall', {}),
        'targets': {
            'monthly_return_target': '>= 0.40',
            'sharpe_target': '>= 3.0',
            'max_drawdown': '<= 0.08'
        },
        'execution': {
            'confidence_threshold_base': 0.68,
            'kelly_fractional': 0.35,
            'max_position': 0.25,
            'portfolio_heat_cap': 0.06
        }
    }
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
    logger.info(f'Exported best settings to {path}')


def main():
    ensure_dirs()
    config = load_config('config_ultimate.yaml') if Path('config_ultimate.yaml').exists() else {}

    # 1) Build dataset
    df, X, y, feature_names, target_name = build_dataset(config)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = time_split(X, y)

    # 2) Train base ensemble
    ensemble, base_metrics = train_base_ensemble(config, df)

    # 3) Bayesian optimize LightGBM
    best_params, best_score = bayes_optimize_lgbm(X_train, y_train, X_val, y_val)

    # 4) Walk-forward backtest
    wf_results = run_walk_forward(df[[*feature_names, target_name]], best_params)

    # 5) Export best settings
    export_best_settings(best_params, wf_results)

    # 6) KPI summary
    overall = wf_results.get('overall', {})
    acc = overall.get('accuracy_mean', 0)
    auc = overall.get('auc_mean', 0)
    win = overall.get('win_rate_mean', 0)
    shp = overall.get('sharpe_proxy_mean', 0)
    logger.info('\n===== KPI SUMMARY (Walk-Forward) =====')
    logger.info(f'Accuracy: {acc:.3f}  AUC: {auc:.3f}  Win: {win:.3f}  SharpeProxy: {shp:.3f}')
    logger.info('Targets: Monthly >= 40%, Sharpe >= 3.0, MaxDD <= 8% (see report)')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f'Auto optimization failed: {e}', exc_info=True)
        sys.exit(1)
