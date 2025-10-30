"""
Baseline models: Logistic Regression and Random Forest.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from typing import Dict, Tuple
import torch
from torch.utils.data import DataLoader


class BaselineModel:
    """Wrapper for sklearn baseline models."""

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
        print(f"Class distribution: {np.bincount(y_train)}")

        # Fit model
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

    def evaluate(self, data_loader: DataLoader, split: str = "test") -> Dict:
        """
        Evaluate the model.

        Args:
            data_loader: Data loader
            split: Split name (for logging)

        Returns:
            Dictionary with metrics
        """
        X, y = self._extract_data(data_loader)

        # Predictions
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)[:, 1]  # Probability of class 1

        # Compute metrics
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "auroc": roc_auc_score(y, y_pred_proba),
            "f1": f1_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
        }

        print(f"\n{split.capitalize()} Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

        return metrics

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

    def feature_importance(self, feature_names: list, top_k: int = 20) -> Dict:
        """
        Get feature importance (for Random Forest).

        Args:
            feature_names: List of feature names
            top_k: Number of top features to return

        Returns:
            Dictionary with top features and their importance
        """
        if self.model_type != "random_forest":
            return {}

        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_k]

        top_features = {
            feature_names[i]: importances[i]
            for i in indices
        }

        print(f"\nTop {top_k} features:")
        for i, (feat, imp) in enumerate(top_features.items(), 1):
            print(f"  {i:2d}. {feat:30s}: {imp:.4f}")

        return top_features

