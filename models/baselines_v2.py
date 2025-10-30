"""
Enhanced baseline models: Logistic Regression, Random Forest, LightGBM, XGBoost.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from typing import Dict, Tuple
import torch
from torch.utils.data import DataLoader

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available")


class BaselineModelV2:
    """Wrapper for sklearn and gradient boosting baseline models."""

    def __init__(self, model_type: str, config: Dict):
        self.model_type = model_type
        self.config = config

        if model_type == "logistic":
            self.model = LogisticRegression(
                C=config["models"]["logistic"]["C"],
                max_iter=config["models"]["logistic"]["max_iter"],
                class_weight=config["models"]["logistic"]["class_weight"],
                random_state=config["seed"],
                n_jobs=-1,
            )
        elif model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=config["models"]["random_forest"]["n_estimators"],
                max_depth=config["models"]["random_forest"]["max_depth"],
                min_samples_split=config["models"]["random_forest"]["min_samples_split"],
                class_weight=config["models"]["random_forest"]["class_weight"],
                random_state=config["seed"],
                n_jobs=config["models"]["random_forest"]["n_jobs"],
            )
        elif model_type == "lightgbm":
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM not installed. Install with: pip install lightgbm")
            
            params = config["models"]["lightgbm"]
            self.model = lgb.LGBMClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                num_leaves=params["num_leaves"],
                min_child_samples=params["min_child_samples"],
                subsample=params["subsample"],
                colsample_bytree=params["colsample_bytree"],
                class_weight=params["class_weight"],
                random_state=config["seed"],
                n_jobs=-1,
                verbose=-1,
            )
        elif model_type == "xgboost":
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost not installed. Install with: pip install xgboost")
            
            params = config["models"]["xgboost"]
            self.model = xgb.XGBClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                min_child_weight=params["min_child_weight"],
                subsample=params["subsample"],
                colsample_bytree=params["colsample_bytree"],
                scale_pos_weight=params["scale_pos_weight"],
                random_state=config["seed"],
                n_jobs=-1,
                verbosity=0,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def train(self, train_loader: DataLoader, val_loader: DataLoader = None) -> Dict:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)

        Returns:
            Dictionary with training metrics
        """
        print(f"\nTraining {self.model_type}...")

        # Extract features and targets from DataLoader
        X_train, y_train = self._extract_data(train_loader)

        print(f"Training samples: {len(X_train)}")
        print(f"Features: {X_train.shape[1]}")
        
        # Check class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"Class distribution: {dict(zip(unique, counts))}")
        imbalance_ratio = counts[1] / counts[0] if len(counts) == 2 else 1.0
        print(f"Imbalance ratio (pos/neg): {imbalance_ratio:.2f}")

        # Update scale_pos_weight for XGBoost if needed
        if self.model_type == "xgboost" and self.config["models"]["xgboost"]["scale_pos_weight"] == 1.0:
            self.model.set_params(scale_pos_weight=counts[0] / counts[1] if len(counts) == 2 else 1.0)

        # Fit model
        if self.model_type in ["lightgbm", "xgboost"] and val_loader is not None:
            # Use early stopping for gradient boosting
            X_val, y_val = self._extract_data(val_loader)
            
            if self.model_type == "lightgbm":
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric="auc",
                    callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
                )
            else:  # xgboost
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric="auc",
                    verbose=False,
                )
        else:
            self.model.fit(X_train, y_train)

        # Evaluate on train
        train_metrics = self.evaluate(train_loader, split="train")

        # Evaluate on val if provided
        val_metrics = {}
        if val_loader is not None:
            val_metrics = self.evaluate(val_loader, split="val")

        return {
            "train": train_metrics,
            "val": val_metrics,
        }

    def evaluate(self, data_loader: DataLoader, split: str = "test", min_confidence: float = 0.0) -> Dict:
        """
        Evaluate the model.

        Args:
            data_loader: Data loader
            split: Split name (for logging)
            min_confidence: Minimum confidence threshold for predictions

        Returns:
            Dictionary with metrics
        """
        X, y = self._extract_data(data_loader)

        # Predictions
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)[:, 1]  # Probability of class 1

        # All predictions
        metrics = self._compute_metrics(y, y_pred, y_pred_proba)

        print(f"\n{split.capitalize()} Metrics (all predictions):")
        self._print_metrics(metrics)

        # High-confidence predictions
        if min_confidence > 0:
            high_conf_mask = (y_pred_proba > min_confidence) | (y_pred_proba < (1 - min_confidence))
            if high_conf_mask.sum() > 0:
                metrics_high_conf = self._compute_metrics(
                    y[high_conf_mask],
                    y_pred[high_conf_mask],
                    y_pred_proba[high_conf_mask]
                )
                print(f"\n{split.capitalize()} Metrics (confidence > {min_confidence}):")
                print(f"  Samples: {high_conf_mask.sum()} / {len(y)} ({high_conf_mask.sum()/len(y)*100:.1f}%)")
                self._print_metrics(metrics_high_conf)
                metrics["high_confidence"] = metrics_high_conf

        return metrics

    def _compute_metrics(self, y_true, y_pred, y_pred_proba) -> Dict:
        """Compute all metrics."""
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "auroc": roc_auc_score(y_true, y_pred_proba),
            "f1": f1_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "directional_accuracy": accuracy_score(y_true, y_pred),  # Same as accuracy for binary
        }

    def _print_metrics(self, metrics: Dict):
        """Print metrics in a formatted way."""
        for metric, value in metrics.items():
            if metric != "high_confidence":
                print(f"  {metric:20s}: {value:.4f}")

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions.

        Args:
            X: Input features

        Returns:
            (predictions, probabilities)
        """
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        return y_pred, y_pred_proba

    def _extract_data(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features and targets from DataLoader."""
        features_list = []
        targets_list = []

        for features, targets in data_loader:
            # If features are 3D (sequences), flatten to 2D by taking last timestep
            if len(features.shape) == 3:
                features = features[:, -1, :]  # Take last timestep

            features_list.append(features.numpy())
            targets_list.append(targets.numpy())

        X = np.vstack(features_list)
        y = np.concatenate(targets_list)

        return X, y

    def feature_importance(self, feature_names: list, top_k: int = 30) -> Dict:
        """
        Get feature importance.

        Args:
            feature_names: List of feature names
            top_k: Number of top features to return

        Returns:
            Dictionary with top features and their importance
        """
        if self.model_type == "logistic":
            # For logistic regression, use coefficients
            importances = np.abs(self.model.coef_[0])
        elif self.model_type in ["random_forest", "lightgbm", "xgboost"]:
            importances = self.model.feature_importances_
        else:
            return {}

        indices = np.argsort(importances)[::-1][:top_k]

        top_features = {
            feature_names[i]: importances[i]
            for i in indices
        }

        print(f"\nTop {top_k} features ({self.model_type}):")
        for i, (feat, imp) in enumerate(top_features.items(), 1):
            print(f"  {i:2d}. {feat:40s}: {imp:.4f}")

        return top_features

