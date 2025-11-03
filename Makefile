.PHONY: all fetch train eval ui clean test

# Default target
all: fetch train eval

# Download data (30 days, 1-minute bars, public Binance API)
fetch:
	@echo "📥 Fetching data from Binance..."
	python data_manager_worldclass.py --days 30
	@echo "✅ Data fetch complete"

# Train models (per config.yaml: baselines + ensemble)
train:
	@echo "🤖 Training models..."
	python train_ultimate.py
	@echo "✅ Training complete"

# Evaluate (prints metrics; saves plots to results/)
eval:
	@echo "📊 Evaluating strategy..."
	python validate_strategy.py
	@echo "✅ Evaluation complete. Check results/ directory"

# Launch dashboard
ui:
	@echo "🚀 Starting Streamlit dashboard..."
	streamlit run dashboard_streamlit.py

# Clean cache and logs
clean:
	@echo "🧹 Cleaning cache and logs..."
	rm -rf logs/*.parquet
	rm -rf logs/*.csv
	rm -rf wandb/
	@echo "✅ Clean complete"

# Run tests
test:
	@echo "🧪 Running tests..."
	pytest tests/ -v
	@echo "✅ Tests complete"

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Installation complete"

# Quick test (5-minute paper trading)
quick-test:
	@echo "⚡ Running quick test (5 minutes)..."
	python run_trading_with_report.py --duration 5 --interval 20
	@echo "✅ Quick test complete"

# Full backtest (30-day walk-forward)
backtest:
	@echo "📈 Running full backtest..."
	python validate_strategy.py --train-days 90 --test-days 7 --step-days 7
	@echo "✅ Backtest complete"

# Generate all results/ plots
results:
	@echo "Generating results plots..."
	python generate_results_plots.py
	@echo "Results generated in results/ directory"

help:
	@echo "Available commands:"
	@echo "  make all          - Run full pipeline (fetch → train → eval)"
	@echo "  make fetch        - Download 30 days of 1-minute data"
	@echo "  make train        - Train all models"
	@echo "  make eval         - Evaluate and generate results/"
	@echo "  make ui           - Launch Streamlit dashboard"
	@echo "  make test         - Run unit tests"
	@echo "  make clean        - Clean cache and logs"
	@echo "  make quick-test   - 5-minute paper trading test"
	@echo "  make backtest     - Full 30-day walk-forward backtest"
	@echo "  make results      - Generate all plots in results/"

