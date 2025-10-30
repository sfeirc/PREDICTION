"""
Enhanced feature engineering for crypto price prediction.

New features:
- Market microstructure (order book imbalance, spread, depth)
- Time-of-day features (hour, minute, day-of-week)
- Cross-asset features (BTC vs ETH correlation)
- Event detection (volume spikes, volatility spikes)
- Multi-horizon targets
- Data quality improvements (outlier removal, missing data handling)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from data_fetcher import BinanceDataFetcher


class FeatureEngineV2:
    """Enhanced feature engineering with market microstructure and cross-asset features."""

    def __init__(self, config: Dict):
        self.config = config
        self.returns_horizons = config["features"]["returns_horizons"]
        self.vol_windows = config["features"]["volatility_windows"]
        self.volume_windows = config["features"]["volume_windows"]
        self.use_vwap = config["features"]["use_vwap"]
        self.use_rsi = config["features"]["use_rsi"]
        self.use_bollinger = config["features"]["use_bollinger"]
        self.use_orderbook = config["features"].get("use_orderbook", False)
        self.use_time_features = config["features"].get("use_time_features", True)
        self.use_cross_asset = config["features"].get("use_cross_asset", False)
        self.event_based_sampling = config["features"].get("event_based_sampling", False)

    def create_features(self, df: pd.DataFrame, cross_asset_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Create all features from OHLCV data.

        Args:
            df: DataFrame with OHLCV columns
            cross_asset_df: Optional cross-asset DataFrame (e.g., ETH for BTC prediction)

        Returns:
            DataFrame with features
        """
        print("Creating enhanced features...")

        # Data quality: handle missing values and outliers
        features = self._clean_data(df.copy())

        # Price-based features
        features = self._add_returns(features)

        # Volatility features (including regime detection)
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

        # Time-of-day features
        if self.use_time_features:
            features = self._add_time_features(features)

        # Cross-asset features
        if self.use_cross_asset and cross_asset_df is not None:
            features = self._add_cross_asset_features(features, cross_asset_df)

        # Market microstructure (order book) - Note: This would require real-time data
        # For historical data, we'll add synthetic spread/depth features
        if self.use_orderbook:
            features = self._add_microstructure_features(features)

        # Event detection flags
        if self.event_based_sampling:
            features = self._add_event_flags(features)

        print(f"Created {len(features.columns)} total columns (including raw data)")

        return features

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data: handle missing values and outliers."""
        print("  Cleaning data...")

        # Forward fill price data
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            if col in df.columns:
                df[col] = df[col].ffill()

        # Fill volume with 0
        if "volume" in df.columns:
            df["volume"] = df["volume"].fillna(0)

        # Remove extreme outliers in returns (> 10 sigma)
        if len(df) > 100:
            returns = df["close"].pct_change()
            mean_ret = returns.mean()
            std_ret = returns.std()
            outlier_mask = (returns - mean_ret).abs() > 10 * std_ret
            n_outliers = outlier_mask.sum()
            if n_outliers > 0:
                print(f"    Found {n_outliers} outliers (>10σ), capping them...")
                # Cap outliers instead of removing
                df.loc[outlier_mask, "close"] = df.loc[outlier_mask, "close"].shift(1)

        return df

    def _add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add multi-horizon returns."""
        print("  Adding returns...")
        for horizon in self.returns_horizons:
            # Log returns
            df[f"return_{horizon}m"] = np.log(df["close"] / df["close"].shift(horizon))

            # Normalized by volatility (z-score)
            rolling_std = df[f"return_{horizon}m"].rolling(window=max(30, horizon)).std()
            df[f"return_{horizon}m_norm"] = df[f"return_{horizon}m"] / (rolling_std + 1e-8)

        return df

    def _add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add realized volatility features including regime detection."""
        print("  Adding volatility features...")
        log_returns = np.log(df["close"] / df["close"].shift(1))

        for window in self.vol_windows:
            # Rolling standard deviation (realized volatility)
            df[f"volatility_{window}m"] = log_returns.rolling(window=window).std()

            # Parkinson volatility (high-low range)
            df[f"parkinson_vol_{window}m"] = np.sqrt(
                ((np.log(df["high"] / df["low"]) ** 2) / (4 * np.log(2))).rolling(window=window).mean()
            )

            # Volatility of volatility
            df[f"vol_of_vol_{window}m"] = df[f"volatility_{window}m"].rolling(window=window).std()

        # Volatility regime (low/mid/high) - very useful!
        if 60 in self.vol_windows or 240 in self.vol_windows:
            vol_col = f"volatility_240m" if 240 in self.vol_windows else "volatility_60m"
            if vol_col in df.columns:
                low_thresh = df[vol_col].quantile(0.33)
                high_thresh = df[vol_col].quantile(0.67)
                df["vol_regime_low"] = (df[vol_col] <= low_thresh).astype(int)
                df["vol_regime_high"] = (df[vol_col] >= high_thresh).astype(int)

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        print("  Adding volume features...")
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

            # Volume std (volatility of volume)
            df[f"volume_std_{window}m"] = rolling_vol.std()

        return df

    def _add_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add VWAP (Volume Weighted Average Price) features."""
        print("  Adding VWAP features...")
        typical_price = (df["high"] + df["low"] + df["close"]) / 3

        for window in [5, 15, 30, 60]:
            vwap = (typical_price * df["volume"]).rolling(window=window).sum() / (
                df["volume"].rolling(window=window).sum() + 1e-8
            )
            df[f"vwap_{window}m"] = vwap
            df[f"vwap_deviation_{window}m"] = (df["close"] - vwap) / (vwap + 1e-8)

        return df

    def _add_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add RSI (Relative Strength Index)."""
        print("  Adding RSI...")
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / (avg_loss + 1e-8)
        df["rsi"] = 100 - (100 / (1 + rs))
        df["rsi_momentum"] = df["rsi"].diff(5)

        # RSI overbought/oversold flags
        df["rsi_overbought"] = (df["rsi"] > 70).astype(int)
        df["rsi_oversold"] = (df["rsi"] < 30).astype(int)

        return df

    def _add_bollinger(self, df: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
        """Add Bollinger Bands features."""
        print("  Adding Bollinger Bands...")
        sma = df["close"].rolling(window=period).mean()
        std = df["close"].rolling(window=period).std()

        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)

        df["bb_middle"] = sma
        df["bb_upper"] = upper_band
        df["bb_lower"] = lower_band
        df["bb_position"] = (df["close"] - lower_band) / (upper_band - lower_band + 1e-8)
        df["bb_bandwidth"] = (upper_band - lower_band) / (sma + 1e-8)

        # Bollinger squeeze (low volatility)
        df["bb_squeeze"] = (df["bb_bandwidth"] < df["bb_bandwidth"].rolling(window=100).quantile(0.2)).astype(int)

        return df

    def _add_volume_imbalance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume imbalance features (buy vs sell pressure)."""
        print("  Adding volume imbalance...")
        df["taker_buy_ratio"] = df["taker_buy_base"] / (df["volume"] + 1e-8)

        for window in [5, 15, 60]:
            df[f"taker_buy_ratio_ma_{window}m"] = df["taker_buy_ratio"].rolling(window=window).mean()

        df["buy_pressure"] = df["taker_buy_ratio"] * df["volume"]
        df["buy_pressure_ma_5m"] = df["buy_pressure"].rolling(window=5).mean()

        # Buy/sell imbalance momentum
        df["buy_imbalance_momentum"] = df["taker_buy_ratio"].diff(5)

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-of-day cyclical features."""
        print("  Adding time features...")

        # Extract time components
        df["hour"] = df.index.hour
        df["minute"] = df.index.minute
        df["day_of_week"] = df.index.dayofweek

        # Cyclical encoding (sine/cosine)
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
        df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        # Trading session flags (rough approximations)
        # US session: 13:30-20:00 UTC
        # EU session: 07:00-15:30 UTC
        # Asia session: 00:00-08:00 UTC
        df["us_session"] = ((df["hour"] >= 13) & (df["hour"] < 20)).astype(int)
        df["eu_session"] = ((df["hour"] >= 7) & (df["hour"] < 16)).astype(int)
        df["asia_session"] = (df["hour"] < 8).astype(int)

        return df

    def _add_cross_asset_features(self, df: pd.DataFrame, cross_asset_df: pd.DataFrame) -> pd.DataFrame:
        """Add cross-asset features (e.g., ETH for BTC prediction)."""
        print("  Adding cross-asset features...")

        # Align cross-asset data with main data
        cross_asset_df = cross_asset_df.reindex(df.index, method="ffill")

        # Cross-asset returns
        for horizon in [1, 3, 5, 15]:
            cross_returns = np.log(cross_asset_df["close"] / cross_asset_df["close"].shift(horizon))
            df[f"cross_return_{horizon}m"] = cross_returns

        # Spread between assets
        df["cross_spread"] = (df["close"] / cross_asset_df["close"]).pct_change()

        # Rolling correlation
        for window in [30, 60]:
            main_returns = np.log(df["close"] / df["close"].shift(1))
            cross_returns = np.log(cross_asset_df["close"] / cross_asset_df["close"].shift(1))
            df[f"cross_corr_{window}m"] = main_returns.rolling(window=window).corr(cross_returns)

        # Lead-lag: does cross-asset lead main asset?
        cross_returns_1m = np.log(cross_asset_df["close"] / cross_asset_df["close"].shift(1))
        df["cross_lead_1m"] = cross_returns_1m.shift(1)  # Previous minute cross-asset return

        return df

    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market microstructure features (synthetic for historical data).
        
        Note: For live trading, these should be fetched from order book API.
        For historical backtesting, we approximate them.
        """
        print("  Adding microstructure features (synthetic)...")

        # Approximate bid-ask spread using high-low range
        df["spread_proxy"] = (df["high"] - df["low"]) / df["close"]
        df["spread_ma_5m"] = df["spread_proxy"].rolling(window=5).mean()

        # Spread compression (tightening = higher liquidity)
        df["spread_compression"] = df["spread_proxy"] / (df["spread_ma_5m"] + 1e-8)

        # Mid-price vs last trade (using OHLC approximation)
        df["mid_price"] = (df["high"] + df["low"]) / 2
        df["mid_vs_close"] = (df["close"] - df["mid_price"]) / df["mid_price"]

        # Price impact proxy (using volume and price range)
        df["price_impact_proxy"] = (df["high"] - df["low"]) / (df["volume"] + 1e-8)

        return df

    def _add_event_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add event detection flags for interesting timesteps."""
        print("  Adding event flags...")

        thresholds = self.config["features"].get("event_thresholds", {})

        # Volume spike event
        if "volume_spike_5m" in df.columns:
            vol_threshold = thresholds.get("volume_spike", 2.0)
            df["event_volume_spike"] = (df["volume_spike_5m"] > vol_threshold).astype(int)

        # Volatility spike event
        if "volatility_5m" in df.columns:
            vol_ma = df["volatility_5m"].rolling(window=30).mean()
            vol_ratio = df["volatility_5m"] / (vol_ma + 1e-8)
            vol_threshold = thresholds.get("volatility_spike", 1.5)
            df["event_volatility_spike"] = (vol_ratio > vol_threshold).astype(int)

        # Spread compression event
        if "spread_compression" in df.columns:
            spread_threshold = thresholds.get("spread_compression", 0.5)
            df["event_spread_compression"] = (df["spread_compression"] < spread_threshold).astype(int)

        # Aggregate event flag (any event)
        event_cols = [col for col in df.columns if col.startswith("event_")]
        if event_cols:
            df["is_event"] = df[event_cols].max(axis=1)

        return df

    def create_multi_horizon_targets(
        self,
        df: pd.DataFrame,
        horizons: List[int] = [5, 10, 15],
        up_threshold: float = 0.002,
        down_threshold: float = -0.002,
    ) -> pd.DataFrame:
        """
        Create multi-horizon targets for multi-task learning.

        Args:
            df: DataFrame with price data
            horizons: List of forward prediction horizons in minutes
            up_threshold: Threshold for "up" label
            down_threshold: Threshold for "down" label

        Returns:
            DataFrame with target columns for each horizon
        """
        print(f"\nCreating multi-horizon targets: {horizons} minutes")

        for horizon in horizons:
            # Forward return
            df[f"forward_return_{horizon}m"] = (
                df["close"].shift(-horizon) - df["close"]
            ) / df["close"]

            # Binary labels
            df[f"target_{horizon}m"] = np.nan
            df.loc[df[f"forward_return_{horizon}m"] > up_threshold, f"target_{horizon}m"] = 1
            df.loc[df[f"forward_return_{horizon}m"] < down_threshold, f"target_{horizon}m"] = 0

            # Count labels
            n_up = (df[f"target_{horizon}m"] == 1).sum()
            n_down = (df[f"target_{horizon}m"] == 0).sum()
            n_neutral = df[f"target_{horizon}m"].isna().sum()
            n_total = len(df)

            print(f"\n  {horizon}m horizon:")
            print(f"    Up:      {n_up:6d} ({n_up/n_total*100:5.2f}%)")
            print(f"    Down:    {n_down:6d} ({n_down/n_total*100:5.2f}%)")
            print(f"    Neutral: {n_neutral:6d} ({n_neutral/n_total*100:5.2f}%) [dropped]")
            print(f"    Imbalance ratio: {n_up / (n_down + 1):.2f}")

        return df

    def balance_classes(
        self,
        df: pd.DataFrame,
        target_col: str = "target_5m",
        method: str = "downsample",
    ) -> pd.DataFrame:
        """
        Balance classes by downsampling majority class or using weights.

        Args:
            df: DataFrame with target column
            target_col: Target column name
            method: "downsample" or "weights"

        Returns:
            Balanced DataFrame (if downsampling) or original DataFrame with weights column
        """
        print(f"\nBalancing classes using method: {method}")

        # Filter valid targets
        df_valid = df[~df[target_col].isna()].copy()

        n_up = (df_valid[target_col] == 1).sum()
        n_down = (df_valid[target_col] == 0).sum()

        print(f"Before balancing: Up={n_up}, Down={n_down}, Ratio={n_up/n_down:.2f}")

        if method == "downsample":
            # Downsample majority class
            minority_class = 1 if n_up < n_down else 0
            majority_class = 1 - minority_class
            n_minority = min(n_up, n_down)

            # Keep all minority samples
            df_minority = df_valid[df_valid[target_col] == minority_class]

            # Downsample majority
            df_majority = df_valid[df_valid[target_col] == majority_class].sample(
                n=n_minority,
                random_state=42,
            )

            # Combine
            df_balanced = pd.concat([df_minority, df_majority]).sort_index()

            print(f"After downsampling: {len(df_balanced)} samples (50/50 split)")

            return df_balanced

        elif method == "weights":
            # Add sample weights (inverse of class frequency)
            total = n_up + n_down
            df_valid["sample_weight"] = np.where(
                df_valid[target_col] == 1,
                total / (2 * n_up),
                total / (2 * n_down),
            )

            print(f"Added sample weights: up_weight={total/(2*n_up):.2f}, down_weight={total/(2*n_down):.2f}")

            return df_valid

        else:
            raise ValueError(f"Unknown balancing method: {method}")

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature column names (exclude OHLCV and target)."""
        exclude_cols = {
            "open", "high", "low", "close", "volume",
            "quote_volume", "trades", "taker_buy_base", "taker_buy_quote",
            "hour", "minute", "day_of_week",  # Keep cyclical versions only
            "mid_price",  # Internal calculation
        }

        # Exclude all target and forward_return columns
        exclude_patterns = ["target_", "forward_return_", "sample_weight"]

        feature_cols = [
            col for col in df.columns
            if col not in exclude_cols
            and not any(pattern in col for pattern in exclude_patterns)
        ]

        return feature_cols


def main():
    """Test enhanced feature engineering."""
    import yaml
    from data_fetcher import BinanceDataFetcher

    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Fetch data
    fetcher = BinanceDataFetcher(cache_dir=config["data"]["cache_dir"])

    # Fetch BTC
    btc_df = fetcher.fetch_and_cache(
        symbol="BTCUSDT",
        days=config["data"]["days"],
        interval=config["data"]["interval"],
    )

    # Fetch ETH for cross-asset features
    eth_df = None
    if config["features"].get("use_cross_asset"):
        eth_df = fetcher.fetch_and_cache(
            symbol="ETHUSDT",
            days=config["data"]["days"],
            interval=config["data"]["interval"],
        )

    # Create features
    feature_engine = FeatureEngineV2(config)
    features_df = feature_engine.create_features(btc_df, cross_asset_df=eth_df)

    # Create multi-horizon targets
    features_df = feature_engine.create_multi_horizon_targets(
        features_df,
        horizons=config["target"]["horizons"],
        up_threshold=config["target"]["up_threshold"],
        down_threshold=config["target"]["down_threshold"],
    )

    # Balance classes
    if config["target"].get("balance_classes"):
        features_df = feature_engine.balance_classes(
            features_df,
            target_col="target_5m",
            method=config["target"]["balancing_method"],
        )

    # Get feature columns
    feature_cols = feature_engine.get_feature_columns(features_df)

    print("\n" + "=" * 80)
    print("Enhanced Feature Engineering Summary")
    print("=" * 80)
    print(f"Total rows: {len(features_df)}")
    print(f"Total features: {len(feature_cols)}")
    print(f"\nFeature categories:")

    # Group features by category
    categories = {
        "Returns": [f for f in feature_cols if "return" in f and "forward" not in f],
        "Volatility": [f for f in feature_cols if "vol" in f or "parkinson" in f],
        "Volume": [f for f in feature_cols if "volume" in f],
        "VWAP": [f for f in feature_cols if "vwap" in f],
        "Indicators": [f for f in feature_cols if any(x in f for x in ["rsi", "bb_"])],
        "Imbalance": [f for f in feature_cols if "taker" in f or "buy" in f or "pressure" in f],
        "Time": [f for f in feature_cols if any(x in f for x in ["hour", "minute", "dow", "session"])],
        "Cross-asset": [f for f in feature_cols if "cross" in f],
        "Microstructure": [f for f in feature_cols if any(x in f for x in ["spread", "mid_", "impact"])],
        "Events": [f for f in feature_cols if "event" in f or "is_event" in f],
        "Regime": [f for f in feature_cols if "regime" in f],
    }

    for category, feats in categories.items():
        if feats:
            print(f"  {category:15s}: {len(feats):3d} features")

    # Save processed data
    output_path = f"{config['data']['processed_dir']}/btcusdt_features_v2.parquet"
    features_df.to_parquet(output_path)
    print(f"\nSaved enhanced features to: {output_path}")


if __name__ == "__main__":
    main()

