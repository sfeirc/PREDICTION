# Files to Clean Up (Unused/Old Versions)

## Test/Debug Scripts (Can be removed - only used for development)
- `test_1hour.py` - Old test script
- `test_pipeline.py` - Old pipeline test
- `test_worldclass_simple.py` - Simple test
- `test_complete_system.py` - System test (maybe keep for CI?)
- `test_and_report.py` - Test with reporting
- `test_confidence_boost.py` - Confidence boost test
- `quick_debug.py` - Debug script
- `diagnose_confidence.py` - Diagnostic script
- `analyze_predictions.py` - Analysis script
- `verify_setup.py` - Setup verification (can keep if useful)

## Optimization/Finding Scripts (One-off utilities)
- `find_optimal_confidence.py` - One-time optimization
- `find_optimal_tradeoff.py` - One-time optimization
- `maximize_profits.py` - Optimization script
- `maximize_trades_ultimate.py` - Optimization script
- `fetch_max_data.py` - Data fetching utility
- `fetch_max_data_unbalanced.py` - Data fetching utility

## Old Training Scripts (Multiple versions - keep only latest)
- `train.py` - **OLD VERSION** - Remove (use train_ultimate.py)
- `train_v2.py` - **OLD VERSION** - Remove
- `train_profitable.py` - **OLD VERSION** - Remove
- `train_simple_profitable.py` - **OLD VERSION** - Remove
- **KEEP:** `train_ultimate.py` - Latest version

## Old Feature Engineering (Multiple versions)
- `feature_engineering.py` - **OLD VERSION** - Remove (use feature_engine_worldclass.py)
- `feature_engineering_v2.py` - **OLD VERSION** - Remove
- `integrate_ultimate_features.py` - Integration script (maybe obsolete)
- **KEEP:** `feature_engine_worldclass.py` - Latest version
- **KEEP:** `features_ultimate.py` - Ultimate features (if still used)

## Old Data Fetching
- `data_fetcher.py` - **OLD VERSION** - Remove (use data_manager_worldclass.py)
- **KEEP:** `data_manager_worldclass.py` - Latest version

## Old Models (Check if still used)
- `models/baselines.py` - **OLD VERSION** - Remove (use model_ensemble_worldclass.py)
- `models/baselines_v2.py` - **OLD VERSION** - Remove
- `models/seq.py` - **MAYBE KEEP** - If LSTM/Transformer still used
- **KEEP:** `model_ensemble_worldclass.py` - Latest version

## Old Evaluation
- `evaluate.py` - **OLD VERSION** - Remove (use validate_strategy.py)
- **KEEP:** `validate_strategy.py` - Latest version

## Documentation (Multiple READMEs - consolidate)
- `README_WORLDCLASS.md` - **REMOVE** - Consolidated into README.md
- `QUICKSTART.md` - **REMOVE** - Consolidated
- `QUICK_START_TRADING.md` - **REMOVE** - Consolidated
- `START_HERE_ULTIMATE.md` - **REMOVE** - Consolidated
- `ULTIMATE_COMPLETE.md` - **REMOVE** - Consolidated
- `MISSION_ACCOMPLISHED.md` - **REMOVE** - Consolidated
- `100_PERCENT_COMPLETE.md` - **REMOVE** - Consolidated
- `BUILD_STATUS_REPORT.md` - **REMOVE** - Consolidated
- `ULTIMATE_ENHANCEMENTS.md` - **REMOVE** - Consolidated
- `QUICK_PROFIT_BOOSTERS.md` - **REMOVE** - Consolidated
- `NUCLEAR_BUILD_PLAN.md` - **REMOVE** - Consolidated (roadmap in README)
- `IMPROVEMENTS.md` - **REMOVE** - Consolidated
- `FINAL_SUMMARY.txt` - **REMOVE** - Consolidated
- `WORLDCLASS_COMPLETE.txt` - **REMOVE** - Consolidated
- `IMPLEMENTATION_COMPLETE.txt` - **REMOVE** - Consolidated
- `IMPLEMENTATION_SUMMARY.txt` - **REMOVE** - Consolidated
- `STATUS_READY.md` - **REMOVE** - Consolidated
- `READY_FOR_GITHUB.md` - **REMOVE** - Consolidated
- `GITHUB_SETUP.md` - **REMOVE** - Consolidated
- `NEXT_STEPS.md` - **REMOVE** - Consolidated (roadmap in README)
- `TRADING_STATUS_REPORT.md` - **REMOVE** - Consolidated
- `FIXED_MEMORY_AND_PREDICTIONS.md` - **REMOVE** - Consolidated
- `FIXED_PREDICTIONS_INFO.md` - **REMOVE** - Consolidated
- `ADAPTIVE_OPTIMIZATION_EXPLAINED.md` - **REMOVE** - Consolidated
- `_READ_ME_FIRST.txt` - **REMOVE** - Consolidated
- **KEEP:** `README.md` - Main consolidated README
- **KEEP:** `SECURITY.md` - Security policy
- **KEEP:** `LICENSE` - License file

## Old Config Files
- `config.yaml` - **OLD VERSION** - Remove (use config_ultimate.yaml)
- `config_profitable.yaml` - **OLD VERSION** - Remove
- `config_worldclass.yaml` - **OLD VERSION** - Remove (unless still used)
- **KEEP:** `config_ultimate.yaml` - Latest version

## Utility Scripts (Maybe keep some)
- `quickstart.py` - Quickstart script (maybe keep for convenience?)
- `utils.py` - **KEEP** - Utilities
- `datasets.py` - **MAYBE KEEP** - Dataset utilities
- `losses.py` - **MAYBE KEEP** - Loss functions
- `trading_simulator.py` - **KEEP** - Core component

## Demo/Integration Scripts
- `DEMO_worldclass.py` - **REMOVE** - Demo script
- `worldclass_bot.py` - **MAYBE REMOVE** - Check if used
- `bot_ultimate_integrated.py` - **MAYBE KEEP** - If main integration point

## Other Scripts to Review
- `paper_trade.py` - **MAYBE KEEP** - If used for paper trading
- `run_trading_with_report.py` - **KEEP** - Main trading script
- `run_wf_fast.py` - **MAYBE REMOVE** - Fast walk-forward test
- `auto_optimize.py` - **KEEP** - Bayesian optimization
- `backtest_walkforward.py` - **KEEP** - Core component
- `backtest_engine_worldclass.py` - **MAYBE REMOVE** - If duplicate of backtest_walkforward.py

## Batch Files (Windows-specific)
- `DAILY_RUN.bat` - **MAYBE KEEP** - Convenience script
- `RUN_1HOUR_TEST.bat` - **REMOVE** - Old test
- `RUN_COMPLETE_TEST.bat` - **REMOVE** - Old test
- `RUN_ULTIMATE_TEST.bat` - **REMOVE** - Old test
- `START_DASHBOARD.bat` - **MAYBE KEEP** - Convenience script
- `PUSH_TO_GITHUB.bat` - **MAYBE KEEP** - Convenience script
- `install_dependencies.bat` - **MAYBE KEEP** - Setup script

## Directories to Clean
- `catboost_info/` - **REMOVE** - CatBoost training artifacts
- `wandb/` - **REMOVE** - W&B logs (or add to .gitignore)
- `checkpoints/` - **MAYBE CLEAN** - Old model checkpoints
- `logs/` - **MAYBE CLEAN** - Keep recent logs, remove old

## Files Generated by Scripts (Add to .gitignore)
- `generate_results_plots.py` - **KEEP** - Script to generate plots
- `results/*.png` - **KEEP** - Generated plots (already in results/)
- `results/*.csv` - **KEEP** - Generated metrics

---

## Summary

### Definitely Remove (~40 files):
- Old versions of training/feature/data scripts
- Multiple documentation files (consolidated into README.md)
- Test/debug scripts (development only)
- One-off optimization scripts

### Review/Maybe Remove (~10 files):
- Demo scripts
- Duplicate backtest engines
- Old batch files
- Integration scripts

### Keep (Core System):
- `train_ultimate.py`
- `validate_strategy.py`
- `run_trading_with_report.py`
- `data_manager_worldclass.py`
- `feature_engine_worldclass.py`
- `model_ensemble_worldclass.py`
- `risk_manager_worldclass.py`
- `trading_simulator.py`
- `backtest_walkforward.py`
- `auto_optimize.py`
- `bayesian_optimization.py`
- `dashboard_streamlit.py`
- `config_ultimate.yaml`
- `README.md` (consolidated)
- `requirements.txt`
- `Makefile`
- `LICENSE`
- `SECURITY.md`

