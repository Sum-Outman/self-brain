@echo off
echo Starting GitHub push process...
echo.

REM 检查网络连接
echo Testing GitHub connectivity...
ping -n 1 github.com >nul 2>&1
if %errorlevel% neq 0 (
    echo Network connection failed. Please check your internet connection.
    pause
    exit /b 1
)

echo Network connection OK.
echo.

REM 设置远程仓库（HTTPS方式）
echo Setting up remote repository...
git remote set-url origin https://github.com/Sum-Outman/self-brain.git

REM 获取最新更改
echo Fetching latest changes...
git fetch origin

REM 推送主分支
echo Pushing to main branch...
git push origin main

if %errorlevel% neq 0 (
    echo HTTPS push failed, trying SSH...
    git remote set-url origin git@github.com:Sum-Outman/self-brain.git
    git push origin main
)

if %errorlevel% neq 0 (
    echo Both HTTPS and SSH push failed.
    echo Please check your GitHub credentials and network settings.
    echo You may need to:
    echo 1. Check GitHub token settings
    echo 2. Verify SSH key configuration
    echo 3. Use GitHub Desktop as alternative
) else (
    echo Successfully pushed to GitHub!
)

pause