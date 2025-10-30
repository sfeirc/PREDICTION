@echo off
echo ================================================================================
echo 🚀 PUSHING ULTIMATE TRADING BOT TO GITHUB
echo ================================================================================
echo.
echo Repository: https://github.com/sfeirc/PREDICTION
echo.
echo This will:
echo   1. Initialize Git repository
echo   2. Add all files
echo   3. Create initial commit
echo   4. Push to GitHub
echo.
pause
echo.

echo 📋 Step 1: Initializing Git repository...
git init
if errorlevel 1 (
    echo ❌ Git not found! Install Git first: https://git-scm.com/downloads
    pause
    exit /b 1
)
echo ✅ Git initialized
echo.

echo 📦 Step 2: Adding all files...
git add .
echo ✅ Files added
echo.

echo 💾 Step 3: Creating commit...
git commit -m "🏆 Initial commit: Ultimate Trading Bot - Complete system with 18 components, 5800+ lines of code"
echo ✅ Commit created
echo.

echo 🔗 Step 4: Adding remote repository...
git remote add origin https://github.com/sfeirc/PREDICTION.git
if errorlevel 1 (
    echo Remote already exists, updating URL...
    git remote set-url origin https://github.com/sfeirc/PREDICTION.git
)
echo ✅ Remote added
echo.

echo 🌿 Step 5: Setting branch to main...
git branch -M main
echo ✅ Branch set to main
echo.

echo 🚀 Step 6: Pushing to GitHub...
echo.
echo ⚠️  You may be prompted for:
echo    Username: sfeirc
echo    Password: [Your GitHub Personal Access Token]
echo.
echo Don't have a token? Get one here:
echo https://github.com/settings/tokens
echo.
git push -u origin main

if errorlevel 1 (
    echo.
    echo ❌ Push failed!
    echo.
    echo Possible reasons:
    echo   1. Need to authenticate (use Personal Access Token)
    echo   2. Repository doesn't exist
    echo   3. Network issues
    echo.
    echo Solutions:
    echo   1. Get token: https://github.com/settings/tokens
    echo   2. Use GitHub CLI: gh auth login
    echo   3. Check internet connection
    echo.
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo ✅ SUCCESS! Your bot is now on GitHub!
echo ================================================================================
echo.
echo 🌐 View it at: https://github.com/sfeirc/PREDICTION
echo.
echo 📋 Next steps:
echo    1. Visit the repository
echo    2. Add topics (trading-bot, cryptocurrency, machine-learning, etc.)
echo    3. Add description
echo    4. Create first release (v1.0.0)
echo    5. Share with the community!
echo.
echo 🎉 Congratulations!
echo.
pause

