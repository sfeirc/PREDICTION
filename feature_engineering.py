"""
Feature engineering for crypto price prediction.

Features:
- Multi-horizon returns (1m, 3m, 5m, 15m)
- Realized volatility (rolling std)
- Volume features (rolling volume, volume spikes)
- VWAP deviation
- Technical indicators (RSI, Bollinger Bands)
- Volume imbalance (taker buy vs total volume)
"""

import numpy as np
import pandas as pd
from typing import Dict, List


class FeatureEngine:
    """Engineer features from OHLCV data."""

    def __init__(self, config: Dict):
        self.config = config
        self.returns_horizons = config["features"]["returns_horizons"]
        self.vol_windows = config["features"]["volatility_windows"]
        self.volume_windows = config["features"]["volume_windows"]
        self.use_vwap = config["features"]["use_vwap"]
        self.use_rsi = config["features"]["use_rsi"]
        self.use_bollinger = config["features"]["use_bollinger"]

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features from OHLCV data.

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            DataFrame with features
        """
        print("Creating features...")

        # Make a copy to avoid modifying original
        features = df.copy()

        # Price-based features
        features = self._add_returns(features)

        # Volatility features
        features = self._add_volatility(features)

        # Volume features
        features = self._add_volume_features(features)

        # VWAP
        if self.use_vwap:
            features = self._add_vwap(features)

        # Technical indicators
        if self.use_rsi:
            features = self._add_rsi(features)

        if self.use_bollinger:
            features = self._add_bollinger(features)

        # Volume imbalance
        features = self._add_volume_imbalance(features)

        print(f"Created {len(features.columns)} features")

        return features

    def _add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add multi-horizon returns."""
        for horizon in self.returns_horizons:
            # Log returns
            df[f"return_{horizon}m"] = np.log(df["close"] / df["close"].shift(horizon))

            # Normalized by volatility (z-score)
            rolling_std = df[f"return_{horizon}m"].rolling(window=30).std()
            df[f"return_{horizon}m_norm"] = df[f"return_{horizon}m"] / (rolling_std + 1e-8)

        return df

    def _add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add realized volatility features."""
        # Compute log returns for volatility calculation
        log_returns = np.log(df["close"] / df["close"].shift(1))

        for window in self.vol_windows:
            # Rolling standard deviation (realized volatility)
            df[f"volatility_{window}m"] = log_returns.rolling(window=window).std()

            # Parkinson volatility (high-low range)
            df[f"parkinson_vol_{window}m"] = np.sqrt(
                ((np.log(df["high"] / df["low"]) ** 2) / (4 * np.log(2))).rolling(window=window).mean()
            )

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        for window in self.volume_windows:
            # Rolling volume
            rolling_vol = df["volume"].rolling(window=window)
            df[f"volume_ma_{window}m"] = rolling_vol.mean()

            # Volume spike (current volume / mean volume)
            df[f"volume_spike_{window}m"] = df["volume"] / (df[f"volume_ma_{window}m"] + 1e-8)

            # Volume momentum
            df[f"volume_momentum_{window}m"] = (
                df["volume"] - df["volume"].shift(window)
            ) / (df["volume"].shift(window) + 1e-8)

        return df

    def _add_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add VWAP (Volume Weighted Average Price) features."""
        # Typical price
        typical_price = (df["high"] + df["low"] + df["close"]) / 3

        # VWAP over different windows
        for window in [5, 15, 30]:
            # VWAP
            vwap = (typical_price * df["volume"]).rolling(window=window).sum() / (
                df["volume"].rolling(window=window).sum() + 1e-8
            )
            df[f"vwap_{window}m"] = vwap

            # Deviation from VWAP
            df[f"vwap_deviation_{window}m"] = (df["close"] - vwap) / (vwap + 1e-8)

        return df

    def _add_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add RSI (Relative Strength Index)."""
        # Price changes
        delta = df["close"].diff()

        # Gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Average gains and losses
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        # RS and RSI
        rs = avg_gain / (avg_loss + 1e-8)
        df["rsi"] = 100 - (100 / (1 + rs))

        # RSI momentum
        df["rsi_momentum"] = df["rsi"].diff(5)

        return df

    def _add_bollinger(self, df: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
        """Add Bollinger Bands features."""
        # Middle band (SMA)
        sma = df["close"].rolling(window=period).mean()

        # Standard deviation
        std = df["close"].rolling(window=period).std()

        # Upper and lower bands
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)

        # Bollinger features
        df["bb_middle"] = sma
        df["bb_upper"] = upper_band
        df["bb_lower"] = lower_band

        # Position within bands (0 = lower, 1 = upper)
        df["bb_position"] = (df["close"] - lower_band) / (upper_band - lower_band + 1e-8)

        # Bandwidth (volatility indicator)
        df["bb_bandwidth"] = (upper_band - lower_band) / (sma + 1e-8)

        return df

    def _add_volume_imbalance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume imbalance features (buy vs sell pressure)."""
        # Taker buy ratio
        df["taker_buy_ratio"] = df["taker_buy_base"] / (df["volume"] + 1e-8)

        # Rolling taker buy ratio
        for window in [5, 15]:
            df[f"taker_buy_ratio_ma_{window}m"] = df["taker_buy_ratio"].rolling(window=window).mean()

        # Volume-weighted buy pressure
        df["buy_pressure"] = df["taker_buy_ratio"] * df["volume"]
        df["buy_pressure_ma_5m"] = df["buy_pressure"].rolling(window=5).mean()

        return df

    def create_target(
        self,
        df: pd.DataFrame,
        forward_minutes: int = 5,
        up_threshold: float = 0.001,
        down_threshold: float = -0.001,
    ) -> pd.DataFrame:
        """
        Create target labels for prediction.

        Args:
            df: DataFrame with price data
            forward_minutes: How many minutes ahead to predict
            up_threshold: Threshold for "up" label (e.g., 0.001 = +0.1%)
            down_threshold: Threshold for "down" label (e.g., -0.001 = -0.1%)

        Returns:
            DataFrame with 'target' column and 'forward_return' column
        """
        print(f"Creating target: {forward_minutes}m forward return")
        print(f"Up threshold: {up_threshold:.3%}, Down threshold: {down_threshold:.3%}")

        # Forward return
        df["forward_return"] = (df["close"].shift(-forward_minutes) - df["close"]) / df["close"]

        # Binary labels (1 = up, 0 = down, NaN = neutral/drop)
        df["target"] = np.nan
        df.loc[df["forward_return"] > up_threshold, "target"] = 1
        df.loc[df["forward_return"] < down_threshold, "target"] = 0

        # Count labels
        n_up = (df["target"] == 1).sum()
        n_down = (df["target"] == 0).sum()
        n_neutral = df["target"].isna().sum()
        n_total = len(df)

        print(f"Label distribution:")
        print(f"  Up:      {n_up:6d} ({n_up/n_total*100:5.2f}%)")
        print(f"  Down:    {n_down:6d} ({n_down/n_total*100:5.2f}%)")
        print(f"  Neutral: {n_neutral:6d} ({n_neutral/n_total*100:5.2f}%) [dropped]")
        print(f"  Total:   {n_total:6d}")

        return df

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature column names (exclude OHLCV and target)."""
        exclude_cols = {
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
            "target",
            "forward_return",
        }

        feature_cols = [col for col in df.columns if col not in exclude_cols]

        return feature_cols


def main():
    """Test feature engineering."""
    import yaml
    from data_fetcher import BinanceDataFetcher

    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Fetch data
    fetcher = BinanceDataFetcher(cache_dir=config["data"]["cache_dir"])
    df = fetcher.fetch_and_cache(
        symbol=config["data"]["train_symbol"],
        days=config["data"]["days"],
        interval=config["data"]["interval"],
    )

    # Create features
    feature_engine = FeatureEngine(config)
    features_df = feature_engine.create_features(df)

    # Create target
    features_df = feature_engine.create_target(
        features_df,
        forward_minutes=config["target"]["forward_minutes"],
        up_threshold=config["target"]["up_threshold"],
        down_threshold=config["target"]["down_threshold"],
    )

    # Get feature columns
    feature_cols = feature_engine.get_feature_columns(features_df)

    print("\n" + "=" * 60)
    print("Feature Engineering Summary")
    print("=" * 60)
    print(f"Total rows: {len(features_df)}")
    print(f"Total features: {len(feature_cols)}")
    print(f"\nFeature columns:")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i:2d}. {col}")

    print(f"\nMissing values:")
    print(features_df[feature_cols].isnull().sum().sort_values(ascending=False).head(10))

    print(f"\nFirst few rows:")
    print(features_df[feature_cols].head())

    # Save processed data
    output_path = f"{config['data']['processed_dir']}/{config['data']['train_symbol'].lower()}_features.parquet"
    features_df.to_parquet(output_path)
    print(f"\nSaved processed features to: {output_path}")


if __name__ == "__main__":
    main()

