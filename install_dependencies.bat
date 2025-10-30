@echo off
REM Installation script for Windows with SSL certificate workaround

echo ================================================================================
echo Installing Dependencies for Crypto Price Prediction
echo ================================================================================
echo.

REM Upgrade pip first
echo [1/3] Upgrading pip...
python.exe -m pip install --upgrade pip --trusted-host pypi.org --trusted-host files.pythonhosted.org
echo.

REM Install core packages first
echo [2/3] Installing core packages...
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org pandas numpy torch scikit-learn requests tqdm pyyaml matplotlib seaborn
echo.

REM Install remaining packages
echo [3/3] Installing remaining packages...
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org lightgbm xgboost pyarrow imbalanced-learn plotly wandb tensorboard jupyter ipykernel
echo.

echo ================================================================================
echo Installation Complete!
echo ================================================================================
echo.
echo Next steps:
echo   1. python verify_setup.py
echo   2. python data_fetcher.py --symbol BTCUSDT --days 3
echo.
pause

