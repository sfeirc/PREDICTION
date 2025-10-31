"""
Multi-Timeframe Feature Engineering with 100+ Professional Features
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging
from ta import momentum, trend, volatility, volume

logger = logging.getLogger(__name__)


class FeatureEngineWorldClass:
    """
    Professional feature engineering inspired by quantitative hedge funds.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        logger.info("🔧 Feature Engine initialized")
    
    def create_features(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
        """
        Create comprehensive feature set from multi-timeframe data.
        
        Args:
            data: {symbol: {timeframe: DataFrame}}
        
        Returns:
            DataFrame with all features
        """
        logger.info("\n🎨 Creating world-class features...")
        
        primary_pair = self.config['data']['primary_pair']
        primary_data = data[primary_pair]
        
        # Start with 1m data as base
        df = primary_data['1m'].copy()
        
        # Remove duplicate timestamps (keep last)
        if df.index.duplicated().any():
            duplicates = df.index.duplicated().sum()
            logger.info(f"   Removing {duplicates} duplicate timestamps...")
            df = df[~df.index.duplicated(keep='last')]
        
        # Final check - ensure index is unique
        if df.index.duplicated().any():
            df = df.loc[~df.index.duplicated(keep='last')]
        
        logger.info(f"   Base data: {len(df)} rows")
        
        # 1. Price-based features (multi-timeframe)
        logger.info("   Adding price features...")
        df = self._add_price_features(df, primary_data)
        
        # 2. Technical indicators (multi-timeframe)
        logger.info("   Adding technical indicators...")
        df = self._add_technical_indicators(df, primary_data)
        
        # 3. Volume features
        logger.info("   Adding volume features...")
        df = self._add_volume_features(df)
        
        # 4. Volatility features
        logger.info("   Adding volatility features...")
        df = self._add_volatility_features(df, primary_data)
        
        # 5. Cross-asset features
        logger.info("   Adding cross-asset features...")
        df = self._add_cross_asset_features(df, data)
        
        # 6. Market microstructure (orderbook)
        logger.info("   Adding microstructure features...")
        df = self._add_microstructure_features(df)
        
        # 7. Time features
        logger.info("   Adding time features...")
        df = self._add_time_features(df)
        
        # 8. Regime detection
        logger.info("   Adding regime features...")
        df = self._add_regime_features(df)
        
        # 9. Target creation
        logger.info("   Creating targets...")
        df = self._create_targets(df)
        
        # Clean data
        logger.info("   Cleaning data...")
        df = self._clean_data(df)
        
        # Log final feature count
        feature_cols = self.get_feature_columns(df)
        logger.info(f"   Total features: {len(feature_cols)}")
        
        feature_count = len([col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']])
        logger.info(f"\n✅ Created {feature_count} features!")
        
        return df
    
    def _add_price_features(self, df: pd.DataFrame, multi_tf_data: Dict) -> pd.DataFrame:
        """Add price-based features across timeframes."""
        
        # Returns (multiple horizons)
        for h in [1, 3, 5, 10, 15, 30, 60]:
            df[f'return_{h}m'] = df['close'].pct_change(h)
        
        # Log returns
        df['log_return_1m'] = np.log(df['close'] / df['close'].shift(1))
        
        # Price momentum
        for window in [5, 15, 30, 60]:
            df[f'momentum_{window}m'] = df['close'] / df['close'].shift(window) - 1
        
        # Higher timeframe trends
        # FIXED: Use reindex instead of join to avoid memory issues
        for tf in ['15m', '1h', '4h']:
            if tf in multi_tf_data:
                tf_df = multi_tf_data[tf]
                # Remove duplicates from higher timeframe data
                if tf_df.index.duplicated().any():
                    tf_df = tf_df[~tf_df.index.duplicated(keep='last')]
                
                # Reindex to match df index (faster and uses less memory)
                tf_return = tf_df['close'].pct_change()
                # Ensure index is unique before reindexing
                if tf_return.index.duplicated().any():
                    tf_return = tf_return[~tf_return.index.duplicated(keep='last')]
                
                # Use unique index first, then reindex to original (handles duplicates)
                if df.index.duplicated().any():
                    df_unique = df.index.unique()
                    reindexed = tf_return.reindex(df_unique, method='ffill')
                    # Map back to original index
                    df[f'return_{tf}_htf'] = reindexed.reindex(df.index, method='ffill')
                else:
                    df[f'return_{tf}_htf'] = tf_return.reindex(df.index, method='ffill')
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame, multi_tf_data: Dict) -> pd.DataFrame:
        """Add comprehensive technical indicators."""
        
        # RSI
        df['rsi_14'] = momentum.RSIIndicator(df['close'], window=14).rsi()
        df['rsi_7'] = momentum.RSIIndicator(df['close'], window=7).rsi()
        
        # Stochastic RSI
        stoch_rsi = momentum.StochRSIIndicator(df['close'])
        df['stoch_rsi'] = stoch_rsi.stochrsi()
        df['stoch_rsi_k'] = stoch_rsi.stochrsi_k()
        df['stoch_rsi_d'] = stoch_rsi.stochrsi_d()
        
        # MACD
        macd = trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bbands = volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_high'] = bbands.bollinger_hband()
        df['bb_mid'] = bbands.bollinger_mavg()
        df['bb_low'] = bbands.bollinger_lband()
        df['bb_width'] = bbands.bollinger_wband()
        df['bb_pct'] = bbands.bollinger_pband()
        
        # EMA crossovers
        df['ema_9'] = df['close'].ewm(span=9).mean()
        df['ema_21'] = df['close'].ewm(span=21).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        df['ema_cross_9_21'] = (df['ema_9'] > df['ema_21']).astype(int)
        
        # ATR (Average True Range)
        df['atr'] = volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        
        # ADX (Average Directional Index)
        adx = trend.ADXIndicator(df['high'], df['low'], df['close'])
        df['adx'] = adx.adx()
        
        # Money Flow Index
        df['mfi'] = volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume']).money_flow_index()
        
        # On-Balance Volume
        df['obv'] = volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        df['obv_ema'] = df['obv'].ewm(span=20).mean()
        
        # Chaikin Money Flow
        df['cmf'] = volume.ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume']).chaikin_money_flow()
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        
        # Volume ratios
        for window in [5, 15, 30, 60]:
            df[f'volume_ratio_{window}m'] = df['volume'] / df['volume'].rolling(window).mean()
        
        # Volume momentum
        df['volume_momentum'] = df['volume'].pct_change(5)
        
        # Buy/sell pressure
        df['buy_pressure'] = df['taker_buy_base'] / df['volume']
        df['sell_pressure'] = 1 - df['buy_pressure']
        df['pressure_diff'] = df['buy_pressure'] - df['sell_pressure']
        
        # VWAP
        df['vwap'] = (df['close'] * df['volume']).rolling(60).sum() / df['volume'].rolling(60).sum()
        df['vwap_dist'] = (df['close'] - df['vwap']) / df['vwap']
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame, multi_tf_data: Dict) -> pd.DataFrame:
        """Add volatility features."""
        
        # Realized volatility (multiple windows)
        for window in [5, 30, 60, 240]:  # Skip 15 to avoid conflict
            df[f'volatility_{window}m'] = df['close'].pct_change().rolling(window).std()
        
        # Parkinson volatility (more robust)
        df['parkinson_vol_5m'] = np.sqrt(
            (np.log(df['high'] / df['low']) ** 2).rolling(5).mean() / (4 * np.log(2))
        )
        
        # Volatility of volatility
        df['vol_of_vol'] = df['volatility_30m'].rolling(30).std()
        
        # Higher timeframe volatility (use _htf suffix)
        # FIXED: Use reindex instead of join to avoid memory issues
        for tf in ['15m', '1h', '4h']:
            if tf in multi_tf_data:
                tf_df = multi_tf_data[tf]
                # Remove duplicates from higher timeframe data
                if tf_df.index.duplicated().any():
                    tf_df = tf_df[~tf_df.index.duplicated(keep='last')]
                
                tf_vol = tf_df['close'].pct_change().rolling(20).std()
                # Ensure index is unique before reindexing
                if tf_vol.index.duplicated().any():
                    tf_vol = tf_vol[~tf_vol.index.duplicated(keep='last')]
                
                # Handle duplicate indices by reindexing to unique first
                if df.index.duplicated().any():
                    df_unique = df.index.unique()
                    reindexed = tf_vol.reindex(df_unique, method='ffill')
                    # Map back to original index (handles duplicates)
                    df[f'volatility_{tf}_htf'] = reindexed.reindex(df.index, method='ffill')
                else:
                    df[f'volatility_{tf}_htf'] = tf_vol.reindex(df.index, method='ffill')
        
        return df
    
    def _add_cross_asset_features(self, df: pd.DataFrame, data: Dict) -> pd.DataFrame:
        """Add cross-asset correlation features."""
        
        correlation_pairs = self.config['data']['correlation_pairs']
        
        for pair in correlation_pairs:
            if pair in data:
                pair_df = data[pair]['1m'].copy()
                pair_name = pair.replace('USDT', '').lower()
                
                # Remove duplicates from pair data
                if pair_df.index.duplicated().any():
                    pair_df = pair_df[~pair_df.index.duplicated(keep='last')]
                
                # Merge on index using reindex (more memory efficient)
                # Get common index
                common_idx = df.index.intersection(pair_df.index)
                if len(common_idx) > 60:  # Need enough data for correlation
                    pair_close = pair_df['close'].reindex(common_idx, method='ffill')
                    df_close = df.loc[common_idx, 'close']
                    
                    # Price correlation (compute on common index, then reindex to full index)
                    corr_window = 60
                    price_corr_values = []
                    for i in range(len(common_idx)):
                        start = max(0, i - corr_window + 1)
                        window_close = df_close.iloc[start:i+1]
                        window_pair = pair_close.iloc[start:i+1]
                        if len(window_close) > 10:
                            corr_val = window_close.corr(window_pair)
                            price_corr_values.append(corr_val if not np.isnan(corr_val) else 0)
                        else:
                            price_corr_values.append(0)
                    
                    price_corr_series = pd.Series(price_corr_values, index=common_idx)
                    # Ensure unique index before reindexing
                    if df.index.duplicated().any():
                        df_unique = df.loc[~df.index.duplicated(keep='last')]
                        df[f'corr_{pair_name}'] = price_corr_series.reindex(df_unique.index, fill_value=0).reindex(df.index, method='ffill', fill_value=0)
                    else:
                        df[f'corr_{pair_name}'] = price_corr_series.reindex(df.index, fill_value=0)
                    
                    # Return correlation
                    df_returns = df_close.pct_change()
                    pair_returns = pair_close.pct_change()
                    return_corr_values = []
                    for i in range(len(common_idx)):
                        start = max(0, i - corr_window + 1)
                        window_df_ret = df_returns.iloc[start:i+1]
                        window_pair_ret = pair_returns.iloc[start:i+1]
                        if len(window_df_ret) > 10:
                            corr_val = window_df_ret.corr(window_pair_ret)
                            return_corr_values.append(corr_val if not np.isnan(corr_val) else 0)
                        else:
                            return_corr_values.append(0)
                    
                    return_corr_series = pd.Series(return_corr_values, index=common_idx)
                    # Ensure unique index before reindexing
                    if df.index.duplicated().any():
                        df_unique = df.loc[~df.index.duplicated(keep='last')]
                        df[f'return_corr_{pair_name}'] = return_corr_series.reindex(df_unique.index, fill_value=0).reindex(df.index, method='ffill', fill_value=0)
                        spread_series = ratio.pct_change()
                        df[f'spread_{pair_name}'] = spread_series.reindex(df_unique.index, fill_value=0).reindex(df.index, method='ffill', fill_value=0)
                    else:
                        df[f'return_corr_{pair_name}'] = return_corr_series.reindex(df.index, fill_value=0)
                        # Price spread (ratio change)
                        ratio = df_close / pair_close
                        spread = ratio.pct_change()
                        df[f'spread_{pair_name}'] = spread.reindex(df.index, fill_value=0)
                else:
                    # Not enough data - fill with zeros
                    df[f'corr_{pair_name}'] = 0
                    df[f'return_corr_{pair_name}'] = 0
                    df[f'spread_{pair_name}'] = 0
        
        return df
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features (synthetic for now)."""
        
        # Synthetic bid-ask spread estimate
        df['spread_proxy'] = (df['high'] - df['low']) / df['close']
        
        # Price impact proxy
        df['price_impact'] = (df['high'] - df['low']) / (df['volume'] + 1e-8)
        
        # Roll measure (spread estimator)
        df['roll_measure'] = 2 * np.sqrt(-df['close'].diff().rolling(10).cov(df['close'].diff().shift(1)))
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute
        df['day_of_week'] = df.index.dayofweek
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Trading session
        df['session'] = 'off_hours'
        df.loc[(df['hour'] >= 8) & (df['hour'] < 16), 'session'] = 'us_hours'
        df.loc[(df['hour'] >= 0) & (df['hour'] < 8), 'session'] = 'asia_hours'
        df['session'] = df['session'].astype('category').cat.codes
        
        return df
    
    def _add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime detection features."""
        
        # Volatility regime
        vol_60m = df['close'].pct_change().rolling(60).std()
        # Fixed: Use quantile instead of rank for better performance
        vol_percentile = vol_60m.rolling(1440, min_periods=100).quantile(0.5) / vol_60m.rolling(1440, min_periods=100).max()
        vol_percentile = vol_percentile.fillna(0.5)
        
        df['vol_regime'] = 'normal'
        df.loc[vol_percentile < 0.33, 'vol_regime'] = 'low_vol'
        df.loc[vol_percentile > 0.67, 'vol_regime'] = 'high_vol'
        df['vol_regime'] = df['vol_regime'].astype('category').cat.codes
        
        # Trend regime (ADX-based)
        if 'adx' in df.columns:
            df['trend_regime'] = 'ranging'
            df.loc[df['adx'] > 25, 'trend_regime'] = 'trending'
            df['trend_regime'] = df['trend_regime'].astype('category').cat.codes
        
        return df
    
    def _create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create multi-horizon targets."""
        
        horizons = self.config['target']['horizons']
        up_thresh = self.config['target']['up_threshold']
        down_thresh = self.config['target']['down_threshold']
        
        for horizon in horizons:
            # Forward return
            df[f'forward_return_{horizon}m'] = df['close'].shift(-horizon).pct_change(horizon)
            
            # Classification target
            df[f'target_{horizon}m'] = np.nan
            df.loc[df[f'forward_return_{horizon}m'] > up_thresh, f'target_{horizon}m'] = 1
            df.loc[df[f'forward_return_{horizon}m'] < down_thresh, f'target_{horizon}m'] = 0
        
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data."""
        
        # Remove infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill (for higher TF features) - limit to 10 to be safe
        df = df.ffill(limit=10)
        
        # Drop remaining NaNs
        initial_len = len(df)
        df = df.dropna()
        dropped = initial_len - len(df)
        if dropped > 0:
            logger.info(f"   Dropped {dropped} rows with NaN")
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns (exclude targets and raw data)."""
        
        exclude = ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 
                   'trades', 'taker_buy_base', 'taker_buy_quote', 'hour', 'minute', 
                   'day_of_week'] + [col for col in df.columns if 'target_' in col or 'forward_return_' in col]
        
        features = [col for col in df.columns if col not in exclude]
        
        return features

