@echo off
echo 正在推送 Self Brain 项目到 GitHub...
echo.
echo 请确保您已经在 GitHub 上创建了仓库
echo 仓库地址格式: https://github.com/YOUR_USERNAME/self-brain.git
echo.

set /p REPO_URL="请输入您的GitHub仓库URL: "

if "%REPO_URL%"=="" (
    echo 错误: 请输入有效的仓库URL
    pause
    exit /b 1
)

echo.
echo 正在添加远程仓库...
git remote add origin %REPO_URL%

echo.
echo 正在推送到GitHub...
git branch -M main
git push -u origin main

echo.
echo ✅ 推送完成！
echo 项目已成功上传到GitHub
echo.
echo 访问地址: %REPO_URL%
pause