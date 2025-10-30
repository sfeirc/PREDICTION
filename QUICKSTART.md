# Quick Start - Enhanced Crypto Price Prediction

## 🎯 Goal
Predict 5-minute BTC price movements with 55-60% accuracy (vs 50% random baseline).

## ⚡ Fast Track (30 minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

**Note**: If you get errors, install core packages first:
```bash
pip install pandas numpy torch scikit-learn lightgbm xgboost requests tqdm pyyaml matplotlib seaborn
```

### 2. Verify Setup
```bash
python verify_setup.py
```
**Expected**: All checkmarks ✓

### 3. Fetch Small Sample (3 days for testing)
```bash
python data_fetcher.py --symbol BTCUSDT --days 3
python data_fetcher.py --symbol ETHUSDT --days 3
```
**Time**: ~2-3 minutes  
**Size**: ~4,300 rows per symbol

### 4. Create Enhanced Features
```bash
python feature_engineering_v2.py
```
**Time**: ~1 minute  
**Output**: ~50+ features including time-of-day, cross-asset, events

### 5. Quick Test (Optional but Recommended)
```bash
python test_pipeline.py
```
**Time**: ~5 minutes  
**What it does**: Tests all components without full training

### 6. Train LightGBM (Best Model)
```bash
# Coming soon - use original trainer for now
python train.py --model lightgbm --no-wandb
```

## 📊 Expected Results (3 Days Data)

With only 3 days of data (testing):
- **Accuracy**: ~52-54% (slightly above random)
- **AUROC**: ~0.52-0.54
- **Samples**: ~1,500-2,000 after filtering

**This is normal!** Crypto prediction needs more data.

## 🚀 Full Training (Recommended - 4 hours)

### 1. Fetch 30 Days of Data
```bash
python data_fetcher.py --symbol BTCUSDT --days 30
python data_fetcher.py --symbol ETHUSDT --days 30
```
**Time**: ~15-20 minutes  
**Size**: ~43,000 rows per symbol

### 2. Create Features
```bash
python feature_engineering_v2.py
```
**Time**: ~3-5 minutes  
**Output**: `data/processed/btcusdt_features_v2.parquet`

### 3. Train All Models
```bash
# LightGBM (best baseline)
python train.py --model lightgbm

# XGBoost (alternative)
python train.py --model xgboost

# Random Forest (for comparison)
python train.py --model random_forest

# Deep learning (if you have GPU)
python train.py --model transformer
```

### 4. Compare Results
Check logs in `logs/` directory for metrics.

## 📈 With 30 Days Data - Expected Results:

| Model | Accuracy | AUROC | Time |
|-------|----------|-------|------|
| **LightGBM** | 54-57% | 0.55-0.58 | 5 min |
| **XGBoost** | 54-56% | 0.54-0.57 | 10 min |
| **Random Forest** | 53-55% | 0.53-0.56 | 15 min |
| **Transformer** | 55-58% | 0.56-0.59 | 2+ hours |

**High-confidence predictions** (>60% probability):
- **Accuracy**: 60-65% (10% boost!)
- **Only**: 30-40% of predictions

## 🎛️ Tuning for Better Results

### Option 1: Wider Dead Zones (Fewer but Cleaner Labels)
Edit `config.yaml`:
```yaml
target:
  up_threshold: 0.003    # 0.3% instead of 0.2%
  down_threshold: -0.003
```
**Effect**: Fewer samples, but cleaner signal → higher accuracy

### Option 2: Event-Based Training
Edit `config.yaml`:
```yaml
features:
  event_based_sampling: true
```
Then filter events during training:
```python
train_df_events = train_df[train_df['is_event'] == 1]
```
**Effect**: Train only on interesting periods → better signal/noise

### Option 3: High-Volatility Only
Train only on high-volatility samples:
```python
train_df_high_vol = train_df[train_df['vol_regime_high'] == 1]
```
**Effect**: Model excels in specific regimes

## 💰 Trading Simulation

After training, simulate trading:
```python
from trading_simulator import TradingSimulator, backtest_predictions

# Load your predictions (from evaluation)
results = backtest_predictions(
    predictions=y_pred,
    probabilities=y_pred_proba,
    actual_returns=actual_returns,
    config=config,
)

# View results
simulator = TradingSimulator()
simulator.print_results(results)
simulator.plot_results(results, save_path="logs/trading_results.png")
```

**Expected** (with min_confidence=0.6):
- **Win rate**: 52-58%
- **Sharpe ratio**: 0.6-1.0
- **Max drawdown**: 10-20%

## ⚙️ Configuration Reference

Key settings in `config.yaml`:

```yaml
# Target labels (MOST IMPORTANT)
target:
  up_threshold: 0.002        # Wider = cleaner labels
  balance_classes: true       # ALWAYS true for gradient boosting
  
# Features
features:
  use_time_features: true     # Hour, day-of-week
  use_cross_asset: true       # ETH features for BTC prediction
  event_based_sampling: true  # Focus on interesting periods
  
# Training
training:
  loss_type: "focal"          # Good for imbalanced data
  min_confidence: 0.6         # Trade only confident predictions
  
# Models
models:
  lightgbm:
    n_estimators: 200         # More trees = better (but slower)
    learning_rate: 0.05       # Lower = more stable
```

## 🐛 Troubleshooting

### Error: "Module not found"
```bash
pip install -r requirements.txt
```

### Error: "Binance API rate limit"
Wait 1 minute and retry, or use cached data:
```bash
python data_fetcher.py --symbol BTCUSDT --days 30  # no --refresh
```

### Error: "Not enough samples"
Increase days or widen dead zone:
```yaml
target:
  up_threshold: 0.0015  # Narrower threshold = more samples
```

### Low Accuracy (<52%)
1. Check if you have enough data (need 30+ days)
2. Try wider dead zones (0.25% or 0.3%)
3. Enable class balancing
4. Use LightGBM instead of Logistic Regression

### Training Too Slow
1. Reduce `n_estimators` for LightGBM/XGBoost
2. Use fewer features (disable cross_asset or microstructure)
3. Use smaller sample (3-7 days for testing)

## 📊 Monitoring Progress

### During Training
Watch the logs:
```
Train AUROC: 0.5678
Val AUROC: 0.5512    ← Watch this!
```
**Good**: Val AUROC > 0.53  
**Great**: Val AUROC > 0.56  
**Excellent**: Val AUROC > 0.60

### After Training
Check:
1. **Overall metrics**: logs/train_*.log
2. **Per-regime**: Run evaluation with --regime-analysis
3. **Trading simulation**: Check Sharpe ratio and win rate

## 🎯 Success Criteria

### Minimum Viable:
- [x] AUROC > 0.53 (better than random)
- [x] High-confidence accuracy > 55%
- [x] Sharpe ratio > 0.3

### Production-Ready:
- [ ] AUROC > 0.58
- [ ] High-confidence accuracy > 62%
- [ ] Sharpe ratio > 1.0
- [ ] Win rate > 55%
- [ ] Tested on 90+ days of data

## 🔄 Iterative Improvement Loop

1. **Train baseline** (LightGBM, 30 days)
2. **Check feature importance** → Drop low-importance features
3. **Analyze per-regime** → Train separate models per regime?
4. **Adjust thresholds** → Wider dead zone if noisy
5. **Add more data** → 60-90 days
6. **Repeat** until AUROC > 0.56

## 📚 Next Steps

Once you have good results (AUROC > 0.56):
1. **Extend to 90 days** of data
2. **Walk-forward validation** (retrain every week)
3. **Live order book** integration for real microstructure features
4. **Paper trading** for 1 month
5. **Live trading** with small capital

## ⚡ One-Liner Commands

```bash
# Quick test (3 days)
python data_fetcher.py --symbol BTCUSDT --days 3 && python feature_engineering_v2.py && python test_pipeline.py

# Full pipeline (30 days)
python data_fetcher.py --symbol BTCUSDT --days 30 && python data_fetcher.py --symbol ETHUSDT --days 30 && python feature_engineering_v2.py && python train.py --model lightgbm
```

## 💬 Questions?

Check these files:
- `IMPROVEMENTS.md` - Detailed explanation of all improvements
- `README.md` - Original project documentation
- `config.yaml` - All configuration options
- `verify_setup.py` - Verify everything is set up correctly

---

**Ready to start? Run the Fast Track (30 min) above!** 🚀

