# Project Cleanup Summary

## ✅ Completed Tasks

1. **Generated Performance Plots**
   - `results/regime_performance.png` - Regime-aware accuracy
   - `results/ablation.png` - Feature ablation study
   - `results/confidence_curve.png` - Precision vs confidence
   - `results/equity_curve_after_costs.png` - Backtest equity curve
   - `results/metrics.csv` - Full metrics table

2. **Consolidated README**
   - Single comprehensive README.md with all methodology, costs, and features
   - Removed separate docs (methodology.md, costs.md, features.md - content merged)
   - All information in one place

3. **Created Cleanup Script**
   - `cleanup_unused.py` - Removes ~50+ unused files
   - Lists all old versions, test scripts, duplicate docs

## 📋 Next Steps

### To Clean Up Project:

```bash
# Review what will be removed
cat cleanup_unused.py

# Run cleanup (will ask for confirmation)
python cleanup_unused.py
```

### Files That Will Be Removed:

**Old Scripts (~20 files):**
- train.py, train_v2.py, train_profitable.py, etc.
- feature_engineering.py, feature_engineering_v2.py
- test_*.py, quick_debug.py, diagnose_*.py

**Documentation (~20 files):**
- README_WORLDCLASS.md, QUICKSTART.md, etc.
- All consolidated into single README.md

**Directories:**
- catboost_info/ (training artifacts)
- wandb/ (experiment logs - add to .gitignore)

## 📊 Generated Plots

All plots are in `results/` directory and referenced in README.md:
- Regime performance shows accuracy by volatility bucket
- Ablation study shows feature importance
- Confidence curve shows precision vs threshold
- Equity curve shows backtest performance with costs

## 📝 Single README

The README.md now contains:
- Results summary with hard numbers
- Performance plots
- Key results tables
- Rigor & credibility section
- One-command reproducibility (Makefile)
- Complete methodology (temporal validation, regime-aware, costs)
- Feature engineering details
- All consolidated from separate docs

## 🔧 To Generate Plots Anytime:

```bash
make results
# or
python generate_results_plots.py
```

