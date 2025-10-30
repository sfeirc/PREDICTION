@echo off
REM 1-Hour Quick Test Launcher
REM This script runs a complete test of the enhanced pipeline

echo ================================================================================
echo 1-HOUR QUICK TEST - Enhanced Crypto Price Prediction
echo ================================================================================
echo.
echo This will test all improvements in approximately 1 hour:
echo   - Enhanced features (50+)
echo   - LightGBM model
echo   - Trading simulation
echo   - All improvements verified
echo.
echo Estimated time: 45-60 minutes
echo ================================================================================
echo.
echo Press any key to start, or Ctrl+C to cancel...
pause >nul
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Please install Python 3.9 or higher.
    pause
    exit /b 1
)

REM Run the test
python test_1hour.py

echo.
echo ================================================================================
echo Test complete! Check the output above for results.
echo ================================================================================
echo.
pause

