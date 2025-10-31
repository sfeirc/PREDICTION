"""
Run a fast walk-forward backtest using the latest optimized LightGBM params.
Windows: train_days=30, test_days=3, step_days=3 (quick sanity check)
"""

import sys
import io
import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from backtest_walkforward import WalkForwardBacktest
from feature_engine_worldclass import FeatureEngineWorldClass
from data_manager_worldclass import DataManagerWorldClass


def load_config(path: str = 'config_ultimate.yaml') -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def main():
    # Load best params
    best_path = Path('logs/best_settings.yaml')
    if not best_path.exists():
        logger.error('Missing logs/best_settings.yaml - run auto_optimize.py first')
        sys.exit(1)
    best = yaml.safe_load(best_path.read_text(encoding='utf-8')) or {}
    best_params = best.get('best_lightgbm_params', {})
    if not best_params:
        logger.error('best_settings.yaml has no best_lightgbm_params')
        sys.exit(1)

    # Build dataset
    cfg = load_config('config_ultimate.yaml')
    dm = DataManagerWorldClass(cfg)
    data = dm.fetch_all_data()
    fe = FeatureEngineWorldClass(cfg)
    df = fe.create_features(data)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    features = [c for c in df.columns if not c.startswith('target_')]
    tcols = [c for c in df.columns if c.startswith('target_')]
    if not tcols:
        logger.error('No target_* columns present')
        sys.exit(1)
    target = tcols[0]

    from sklearn.metrics import accuracy_score, roc_auc_score
    from lightgbm import LGBMClassifier

    def train_func(train_df: pd.DataFrame):
        X = train_df[features].values
        y = train_df[target].values
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
        model.fit(X, y)
        return model

    def predict_func(model, test_df: pd.DataFrame):
        X = test_df[features].values
        return model.predict_proba(X)[:, 1]

    def evaluate_func(pred_proba, test_df: pd.DataFrame):
        y_true = test_df[target].values
        y_pred = (np.maximum(pred_proba, 1 - pred_proba) > 0.68).astype(int)
        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, pred_proba)
        win = float((y_pred == y_true).mean())
        return {'accuracy': acc, 'auc': auc, 'win_rate': win}

    wf = WalkForwardBacktest(train_days=30, test_days=3, step_days=3, anchored=False)
    results = wf.run(df, train_func, predict_func, evaluate_func)
    overall = results.get('overall', {})

    print('\n=== FAST WALK-FORWARD RESULTS (30/3/3) ===')
    if overall:
        for k, v in sorted(overall.items()):
            if k.endswith('_mean'):
                print(f'{k}: {v:.4f}')
    else:
        print('No windows evaluated (not enough data after cleaning).')


if __name__ == '__main__':
    main()
