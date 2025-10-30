@echo off
echo ================================================================================
echo ULTIMATE TRADING BOT - COMPLETE SYSTEM TEST
echo ================================================================================
echo.

echo Installing missing dependencies...
pip install gym scipy streamlit catboost --trusted-host pypi.org --trusted-host files.pythonhosted.org

echo.
echo ================================================================================
echo Running complete system test...
echo ================================================================================
echo.

python test_complete_system.py

echo.
pause

