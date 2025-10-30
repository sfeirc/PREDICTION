"""
Professional Model Ensemble: LightGBM + XGBoost + CatBoost
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
from pathlib import Path
import joblib

import lightgbm as lgb
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    logger = logging.getLogger(__name__)
    logger.warning("CatBoost not installed - ensemble will use LightGBM + XGBoost only")

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

logger = logging.getLogger(__name__)


class ModelEnsembleWorldClass:
    """
    Professional ensemble of gradient boosting models.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.models = {}
        self.weights = {}
        self.feature_importance = None
        
        self._initialize_models()
        
        logger.info("🤖 Model Ensemble initialized")
    
    def _initialize_models(self):
        """Initialize individual models."""
        
        # LightGBM
        self.models['lightgbm'] = LGBMClassifier(
            **self.config['models']['lightgbm'],
            random_state=self.config['seed'],
            verbose=-1
        )
        
        # XGBoost
        self.models['xgboost'] = XGBClassifier(
            **self.config['models']['xgboost'],
            random_state=self.config['seed'],
            verbosity=0
        )
        
        # CatBoost (if available)
        if HAS_CATBOOST:
            self.models['catboost'] = CatBoostClassifier(
                **self.config['models']['catboost'],
                random_state=self.config['seed'],
                verbose=False
            )
        
        # Set ensemble weights
        self.weights = self.config['models']['ensemble']['weights'].copy()
        if not HAS_CATBOOST and 'catboost' in self.weights:
            # Redistribute catboost weight
            cat_weight = self.weights.pop('catboost')
            for key in self.weights:
                self.weights[key] += cat_weight / len(self.weights)
    
    def train(self, df: pd.DataFrame, config: Dict) -> Dict:
        """
        Train ensemble on data.
        
        Args:
            df: DataFrame with features and targets
            config: Training configuration
        
        Returns:
            Training metrics
        """
        logger.info("\n🎯 Training Ensemble...")
        
        # Get features and target
        from feature_engine_worldclass import FeatureEngineWorldClass
        fe = FeatureEngineWorldClass(config)
        feature_cols = fe.get_feature_columns(df)
        
        primary_horizon = config['target']['primary_horizon']
        target_col = f'target_{primary_horizon}m'
        
        # Filter valid samples
        df_valid = df[~df[target_col].isna()].copy()
        
        X = df_valid[feature_cols].values
        y = df_valid[target_col].values
        
        logger.info(f"   Training samples: {len(X)}")
        logger.info(f"   Features: {len(feature_cols)}")
        logger.info(f"   Class distribution: {np.bincount(y.astype(int))}")
        
        # Time-based split (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        results = {}
        
        # Train each model
        for name, model in self.models.items():
            logger.info(f"\n   Training {name.upper()}...")
            
            if name == 'lightgbm':
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False)]
                )
            elif name == 'xgboost':
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            elif name == 'catboost':
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            else:
                model.fit(X_train, y_train)
            
            # Evaluate
            y_pred_train = model.predict(X_train)
            y_pred_val = model.predict(X_val)
            y_pred_proba_val = model.predict_proba(X_val)[:, 1]
            
            train_acc = accuracy_score(y_train, y_pred_train)
            val_acc = accuracy_score(y_val, y_pred_val)
            val_auc = roc_auc_score(y_val, y_pred_proba_val)
            val_f1 = f1_score(y_val, y_pred_val)
            
            logger.info(f"      Train Acc: {train_acc:.4f}")
            logger.info(f"      Val Acc:   {val_acc:.4f}")
            logger.info(f"      Val AUC:   {val_auc:.4f}")
            logger.info(f"      Val F1:    {val_f1:.4f}")
            
            results[name] = {
                'train_acc': train_acc,
                'val_acc': val_acc,
                'val_auc': val_auc,
                'val_f1': val_f1,
            }
        
        # Ensemble evaluation
        logger.info(f"\n   Evaluating ENSEMBLE...")
        y_pred_ensemble_val = self._ensemble_predict(X_val)
        y_pred_proba_ensemble_val = self._ensemble_predict_proba(X_val)[:, 1]
        
        ensemble_acc = accuracy_score(y_val, y_pred_ensemble_val)
        ensemble_auc = roc_auc_score(y_val, y_pred_proba_ensemble_val)
        ensemble_f1 = f1_score(y_val, y_pred_ensemble_val)
        
        logger.info(f"      Ensemble Acc: {ensemble_acc:.4f}")
        logger.info(f"      Ensemble AUC: {ensemble_auc:.4f}")
        logger.info(f"      Ensemble F1:  {ensemble_f1:.4f}")
        
        results['ensemble'] = {
            'val_acc': ensemble_acc,
            'val_auc': ensemble_auc,
            'val_f1': ensemble_f1,
        }
        
        # Feature importance
        self._calculate_feature_importance(feature_cols)
        
        logger.info("\n✅ Training complete!")
        
        return results
    
    def _ensemble_predict(self, X: np.ndarray) -> np.ndarray:
        """Weighted ensemble prediction."""
        predictions = []
        
        for name, model in self.models.items():
            pred = model.predict(X)
            weight = self.weights.get(name, 0)
            predictions.append(pred * weight)
        
        return (np.sum(predictions, axis=0) > 0.5).astype(int)
    
    def _ensemble_predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Weighted ensemble probability prediction."""
        probas = []
        
        for name, model in self.models.items():
            proba = model.predict_proba(X)
            weight = self.weights.get(name, 0)
            probas.append(proba * weight)
        
        return np.sum(probas, axis=0)
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict using ensemble.
        
        Returns:
            (predictions, probabilities)
        """
        y_pred = self._ensemble_predict(X)
        y_proba = self._ensemble_predict_proba(X)[:, 1]
        
        return y_pred, y_proba
    
    def _calculate_feature_importance(self, feature_names: List[str]):
        """Calculate feature importance from tree-based models."""
        importances = []
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                imp = model.feature_importances_
                importances.append(imp * self.weights[name])
        
        if importances:
            self.feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': np.mean(importances, axis=0)
            }).sort_values('importance', ascending=False)
            
            logger.info(f"\n📊 Top 10 Features:")
            for idx, row in self.feature_importance.head(10).iterrows():
                logger.info(f"      {row['feature']}: {row['importance']:.4f}")
    
    def save_models(self, path: str = "models/saved"):
        """Save trained models to disk."""
        Path(path).mkdir(parents=True, exist_ok=True)
        
        for name, model in self.models.items():
            model_path = Path(path) / f"{name}_model.pkl"
            joblib.dump(model, model_path)
        
        logger.info(f"✅ Models saved to {path}")
    
    def load_models(self, path: str = "models/saved"):
        """Load trained models from disk."""
        for name in self.models.keys():
            model_path = Path(path) / f"{name}_model.pkl"
            if model_path.exists():
                self.models[name] = joblib.load(model_path)
        
        logger.info(f"✅ Models loaded from {path}")

