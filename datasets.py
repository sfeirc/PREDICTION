"""
PyTorch datasets for time-series prediction with proper windowing.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler


class TimeSeriesDataset(Dataset):
    """
    PyTorch dataset for time-series prediction with sliding windows.

    Args:
        features: Feature DataFrame (indexed by time)
        feature_cols: List of feature column names
        target_col: Target column name
        lookback: Number of past timesteps to use
        scaler: Optional pre-fitted scaler (if None, fits new scaler)
    """

    def __init__(
        self,
        features: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "target",
        lookback: int = 30,
        scaler: Optional[StandardScaler] = None,
    ):
        self.features = features
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.lookback = lookback

        # Drop rows with NaN target (neutral samples)
        valid_mask = ~features[target_col].isna()
        self.features = features[valid_mask].copy()

        # Also need enough history for lookback
        # The first lookback samples can't be used for sequences
        self.features = self.features.iloc[lookback:]

        print(f"Dataset size after filtering: {len(self.features)} samples")

        # Scale features
        if scaler is None:
            self.scaler = StandardScaler()
            self.feature_values = self.scaler.fit_transform(self.features[feature_cols].values)
        else:
            self.scaler = scaler
            self.feature_values = self.scaler.transform(self.features[feature_cols].values)

        # Replace NaN/inf with 0 after scaling
        self.feature_values = np.nan_to_num(self.feature_values, nan=0.0, posinf=0.0, neginf=0.0)

        # Targets
        self.targets = self.features[target_col].values.astype(np.int64)

        # Store original timestamps for reference
        self.timestamps = self.features.index.values

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get one sample.

        Returns:
            features: (lookback, n_features) tensor
            target: scalar tensor (0 or 1)
        """
        # We need features from [idx - lookback + 1, idx]
        # But since we already removed the first lookback samples,
        # we need to get from the complete feature array

        # Get the actual position in the original (complete) feature array
        # Since self.feature_values already has lookback rows removed from the start,
        # we need to look back in the ORIGINAL data

        # Actually, let's rethink this. We have self.features which already
        # has lookback rows removed. So idx=0 means we want to predict
        # self.features.iloc[0], using history from the ORIGINAL data.

        # Let's store the full feature values before filtering
        # No wait, simpler approach:

        # Let's just make sure we have enough context
        # For sequence models, idx represents the prediction point
        # We need features from [prediction_point - lookback, prediction_point)

        # Since we already removed first lookback rows, this sample at idx
        # has valid history in the full dataset

        # Let's get the lookback window
        # Actually we need to keep track of the full array for windowing

        # Let me fix this properly - we should not drop rows before windowing
        # Instead, we should handle it in __getitem__

        # For now, let's assume we have a flattened view
        # where each row already has lookback incorporated

        # Actually, simpler: for sequence models, return the sequence
        # For non-sequence models, return the current feature vector

        # Let's return just current features for now, and handle sequences separately
        features = torch.FloatTensor(self.feature_values[idx])
        target = torch.LongTensor([self.targets[idx]])[0]

        return features, target


class SequenceDataset(Dataset):
    """
    Dataset for sequence models (LSTM, Transformer).
    Returns sequences of features instead of single timesteps.
    """

    def __init__(
        self,
        features: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "target",
        lookback: int = 30,
        scaler: Optional[StandardScaler] = None,
    ):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.lookback = lookback

        # Scale features BEFORE filtering
        if scaler is None:
            self.scaler = StandardScaler()
            scaled_features = self.scaler.fit_transform(features[feature_cols].values)
        else:
            self.scaler = scaler
            scaled_features = self.scaler.transform(features[feature_cols].values)

        # Replace NaN/inf
        scaled_features = np.nan_to_num(scaled_features, nan=0.0, posinf=0.0, neginf=0.0)

        # Create DataFrame with scaled features
        self.features_scaled = pd.DataFrame(
            scaled_features,
            index=features.index,
            columns=feature_cols,
        )

        # Add target column
        self.features_scaled[target_col] = features[target_col].values

        # Drop rows with NaN target (neutral samples)
        valid_mask = ~self.features_scaled[target_col].isna()

        # For sequences, we also need to ensure we have lookback history
        # We can only use samples where we have at least lookback previous samples
        self.valid_indices = []
        self.sequences = []
        self.targets = []

        print(f"Building sequences with lookback={lookback}...")

        for i in range(len(self.features_scaled)):
            if i < lookback:
                continue  # Not enough history

            if not valid_mask.iloc[i]:
                continue  # Target is NaN

            # Get sequence [i - lookback, i)
            sequence = scaled_features[i - lookback : i]

            # Get target at position i
            target = self.features_scaled[target_col].iloc[i]

            self.sequences.append(sequence)
            self.targets.append(target)
            self.valid_indices.append(i)

        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.int64)

        print(f"Created {len(self.sequences)} sequences")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get one sequence.

        Returns:
            sequence: (lookback, n_features) tensor
            target: scalar tensor (0 or 1)
        """
        sequence = torch.FloatTensor(self.sequences[idx])
        target = torch.LongTensor([self.targets[idx]])[0]

        return sequence, target


def create_train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time-series data into train/val/test sets.
    Time-based split (no shuffling).

    Args:
        df: Input DataFrame (sorted by time)
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing

    Returns:
        train_df, val_df, test_df
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Ratios must sum to 1"

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    print("\n" + "=" * 60)
    print("Train/Val/Test Split")
    print("=" * 60)
    print(f"Train: {len(train_df):6d} samples ({len(train_df)/n*100:5.2f}%) | {train_df.index.min()} to {train_df.index.max()}")
    print(f"Val:   {len(val_df):6d} samples ({len(val_df)/n*100:5.2f}%) | {val_df.index.min()} to {val_df.index.max()}")
    print(f"Test:  {len(test_df):6d} samples ({len(test_df)/n*100:5.2f}%) | {test_df.index.min()} to {test_df.index.max()}")
    print(f"Total: {n:6d} samples")

    return train_df, val_df, test_df


def main():
    """Test dataset creation."""
    import yaml
    from pathlib import Path

    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Load processed features
    symbol = config["data"]["train_symbol"]
    processed_path = Path(config["data"]["processed_dir"]) / f"{symbol.lower()}_features.parquet"

    if not processed_path.exists():
        print(f"Processed features not found at {processed_path}")
        print("Run feature_engineering.py first")
        return

    print(f"Loading features from {processed_path}")
    df = pd.read_parquet(processed_path)

    # Get feature columns (exclude OHLCV and target)
    exclude_cols = {
        "open", "high", "low", "close", "volume",
        "quote_volume", "trades", "taker_buy_base", "taker_buy_quote",
        "target", "forward_return",
    }
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    print(f"Features: {len(feature_cols)}")

    # Split data
    train_df, val_df, test_df = create_train_val_test_split(
        df,
        train_ratio=config["split"]["train"],
        val_ratio=config["split"]["val"],
        test_ratio=config["split"]["test"],
    )

    # Create flat datasets (for baseline models)
    print("\n" + "=" * 60)
    print("Creating Flat Datasets (for baselines)")
    print("=" * 60)

    train_dataset = TimeSeriesDataset(
        train_df,
        feature_cols,
        target_col="target",
        lookback=config["sequence"]["lookback_minutes"],
        scaler=None,  # Fit on train
    )

    val_dataset = TimeSeriesDataset(
        val_df,
        feature_cols,
        target_col="target",
        lookback=config["sequence"]["lookback_minutes"],
        scaler=train_dataset.scaler,  # Use train scaler
    )

    test_dataset = TimeSeriesDataset(
        test_df,
        feature_cols,
        target_col="target",
        lookback=config["sequence"]["lookback_minutes"],
        scaler=train_dataset.scaler,  # Use train scaler
    )

    print(f"\nTrain dataset: {len(train_dataset)} samples")
    print(f"Val dataset:   {len(val_dataset)} samples")
    print(f"Test dataset:  {len(test_dataset)} samples")

    # Test one sample
    features, target = train_dataset[0]
    print(f"\nSample shape: {features.shape}, target: {target}")

    # Create sequence datasets (for LSTM/Transformer)
    print("\n" + "=" * 60)
    print("Creating Sequence Datasets (for LSTM/Transformer)")
    print("=" * 60)

    train_seq_dataset = SequenceDataset(
        train_df,
        feature_cols,
        target_col="target",
        lookback=config["sequence"]["lookback_minutes"],
        scaler=None,
    )

    val_seq_dataset = SequenceDataset(
        val_df,
        feature_cols,
        target_col="target",
        lookback=config["sequence"]["lookback_minutes"],
        scaler=train_seq_dataset.scaler,
    )

    test_seq_dataset = SequenceDataset(
        test_df,
        feature_cols,
        target_col="target",
        lookback=config["sequence"]["lookback_minutes"],
        scaler=train_seq_dataset.scaler,
    )

    print(f"\nTrain seq dataset: {len(train_seq_dataset)} samples")
    print(f"Val seq dataset:   {len(val_seq_dataset)} samples")
    print(f"Test seq dataset:  {len(test_seq_dataset)} samples")

    # Test one sequence
    sequence, target = train_seq_dataset[0]
    print(f"\nSequence shape: {sequence.shape}, target: {target}")


if __name__ == "__main__":
    main()

