@echo off
setlocal enabledelayedexpansion

echo Self Brain AGI - Simple GitHub Push
 echo ====================================

echo Adding all files to Git...
git add .

echo.
echo Committing changes...
git commit -m "Initial commit: Self Brain AGI System - Complete codebase"

echo.
echo Setting main branch...
git branch -M main

echo.
echo Pushing to GitHub repository...
git push -u origin main --force

if %errorlevel% neq 0 (
    echo.
    echo Error: Push failed.
    echo Trying alternative approach...
    git push origin main --force
    
    if %errorlevel% neq 0 (
        echo.
        echo Error: Still failed. Please try manual push using GitHub Desktop.
        echo Repository URL: https://github.com/Sum-Outman/self-brain
    ) else (
        echo.
        echo Success: All files uploaded to GitHub!
        echo Repository URL: https://github.com/Sum-Outman/self-brain
    )
) else (
    echo.
    echo Success: All files published to GitHub repository!
    echo Repository URL: https://github.com/Sum-Outman/self-brain
)

pause
endlocal