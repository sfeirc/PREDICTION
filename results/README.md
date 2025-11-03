# Results Directory

This directory contains evaluation results from the trading system.

## Expected Files

After running `make eval`, you should see:

- `regime_performance.png` - Accuracy by volatility regime (low/mid/high)
- `ablation.png` - Feature ablation study (delta metrics)
- `confidence_curve.png` - Precision vs confidence threshold
- `equity_curve_after_costs.png` - Backtest equity curve (with transaction costs)
- `metrics.csv` - Full metrics table (per regime, per model)

## Generating Results

```bash
make eval  # Generates all plots and saves to results/
```

Or manually:
```bash
python validate_strategy.py
```

## Notes

- Results are generated with fixed seeds for reproducibility
- All metrics include transaction costs (fees + slippage)
- Plots use consistent color schemes for readability

