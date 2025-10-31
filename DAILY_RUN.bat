@echo off
echo ================================================================================
echo ULTIMATE TRADING BOT - DAILY RUN
echo ================================================================================
echo Timestamp: %date% %time%
set LOG=logs\daily_run_%date:~-4%%date:~3,2%%date:~0,2%.log
echo Logging to %LOG%

echo [1/4] Updating dependencies (skip if already installed)...
pip install -r requirements.txt >> %LOG% 2>&1

echo [2/4] Auto-optimizing (train + BO + walk-forward)...
python auto_optimize.py >> %LOG% 2>&1
if errorlevel 1 (
  echo Auto optimization failed. See %LOG%
  exit /b 1
)

echo [3/4] Starting dashboard (optional; Ctrl+C to stop)...
echo streamlit run dashboard_streamlit.py >> %LOG% 2>&1

echo [4/4] Done. Review logs and best settings at logs\best_settings.yaml
pause
