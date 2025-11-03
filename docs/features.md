# Feature Engineering Documentation

## Feature Categories

### 1. Price/Returns Features

**Multi-Timeframe Returns:**
- `return_1m`, `return_3m`, `return_5m`, `return_10m`, `return_15m`, `return_30m`, `return_60m`
- Formula: `(close[t] - close[t-k]) / close[t-k]`

**Higher-Timeframe Trends:**
- `return_15m_htf`, `return_1h_htf`, `return_4h_htf`
- Computed from higher timeframe, aligned to 1m index via forward-fill

**Momentum:**
- `momentum_5m`, `momentum_15m`, `momentum_30m`, `momentum_60m`
- Formula: `close[t] / close[t-k] - 1`

### 2. Volatility Features

**Realized Volatility:**
- `parkinson_vol_5m`, `parkinson_vol_15m` (Parkinson 1980 estimator)
- `yang_zhang_vol_30m` (Yang-Zhang 2000 estimator, uses OHLC)

**Multi-Timeframe Volatility:**
- `volatility_15m_htf`, `volatility_1h_htf`, `volatility_4h_htf`
- Rolling standard deviation from higher timeframes

**Volatility-of-Volatility:**
- `vol_of_vol` (rolling std of volatility)

### 3. Volume Features

**Volume Profile:**
- `volume_5m`, `volume_15m`, `volume_30m` (rolling volume)
- `volume_spike` (z-score > 2)

**Buy/Sell Pressure:**
- `buy_pressure`, `sell_pressure` (inferred from price-volume)

**VWAP Deviation:**
- `vwap_deviation` (price vs volume-weighted average price)

### 4. Market Microstructure (LOB)

**Spread:**
- `effective_spread` (twice the distance from trade to mid-price)
- `quoted_spread` (bid-ask spread)
- `realized_spread` (effective spread with price impact)

**Order Book Imbalance:**
- `order_imbalance` = `(bid_volume - ask_volume) / (bid_volume + ask_volume)`
- `imbalance_top5` (top-5 levels)
- `imbalance_weighted` (depth-weighted)

**Order Flow Toxicity:**
- `order_flow_toxicity` (VPIN-style, adverse selection proxy)

**Liquidity Score:**
- `liquidity_score` (quote intensity, order arrival rate)

**Microprice:**
- `microprice` (weighted mid-price from order book)

### 5. Time-of-Day Features

**Cyclical Encoding:**
- `minute_of_hour` (0-59, sin/cos encoding)
- `hour_of_day` (0-23, sin/cos encoding)
- `day_of_week` (0-6, sin/cos encoding)

**Session Indicators:**
- `us_session`, `eu_session`, `asia_session` (overlap periods)

### 6. Technical Indicators

**Momentum:**
- `rsi` (RSI-14)
- `stoch_rsi` (Stochastic RSI)

**Trend:**
- `macd`, `macd_signal`, `macd_diff`
- `adx` (Average Directional Index)

**Volatility:**
- `bb_upper`, `bb_middle`, `bb_lower` (Bollinger Bands)
- `bb_width` (band width)

**Volume:**
- `mfi` (Money Flow Index)
- `obv` (On-Balance Volume)
- `cmf` (Chaikin Money Flow)

### 7. Cross-Asset Features

**Correlation:**
- `corr_eth`, `corr_bnb` (price correlation, rolling 30m window)
- `return_corr_eth`, `return_corr_bnb` (return correlation)

**Spread:**
- `spread_eth`, `spread_bnb` (price ratio change)

### 8. Regime Features

**Volatility Regime:**
- `vol_regime` (categorical: low_vol, normal, high_vol)
- `vol_percentile` (24h rolling percentile)

**Trend Regime:**
- `trend_regime` (momentum-based)

## Feature Selection

### Importance Ranking (LightGBM)

Top 10 features by importance:
1. `return_15m_htf` (higher timeframe trend)
2. `return_1h_htf`
3. `return_10m`
4. `volatility_240m`
5. `stoch_rsi`
6. `bb_width`
7. `volatility_15m_htf`
8. `corr_eth`
9. `adx`
10. `parkinson_vol_5m`

### Ablation Results

See [Methodology](methodology.md) for ablation study results.

## Feature Preprocessing

### Missing Values
- Forward-fill for higher timeframe features
- Zero-fill for cross-asset features (when asset unavailable)

### Outliers
- Cap extreme values at 3 standard deviations

### Normalization
- Features not normalized (tree-based models handle raw values)
- Optional: StandardScaler for neural networks (LSTM/Transformer)

## Future Enhancements

1. **Order Flow Features:**
   - Order arrival rate
   - Hidden order detection

2. **Advanced Microstructure:**
   - Kyle's Lambda estimation
   - Effective bid-ask spread (Glosten-Milgrom)

3. **Alternative Data:**
   - Funding rates (perpetual futures)
   - Liquidation data
   - On-chain metrics

