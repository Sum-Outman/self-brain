@echo off
echo Self Brain AGI - GitHub Push Script
echo ====================================

:: Check if git is available
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Git is not installed or not in PATH
    pause
    exit /b 1
)

:: Check current status
echo.
echo Current Git Status:
git status

echo.
echo Adding all changes...
git add .

echo.
echo Committing changes...
git commit -m "feat: Update Self Brain AGI system with enhanced features and documentation"

echo.
echo Attempting to push to GitHub...
git push origin main

if %errorlevel% neq 0 (
    echo.
    echo Push failed. This might be due to:
    echo 1. Network connectivity issues
    echo 2. Authentication problems
    echo 3. Repository access permissions
    echo.
    echo Please check GITHUB_PUSH_GUIDE.md for alternative methods
    echo You can also use GitHub Desktop: https://desktop.github.com/
) else (
    echo.
    echo Successfully pushed to GitHub!
)

pause