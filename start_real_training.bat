@echo off
chcp 65001 >nul
echo 🎯 启动A管理模型实时训练系统
setlocal enabledelayedexpansion

:: 设置颜色
set "RED=[91m"
set "GREEN=[92m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "RESET=[0m"

:: 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo %RED%错误: Python未安装或未添加到PATH%RESET%
    pause
    exit /b 1
)

:: 检查并安装依赖
echo %BLUE%检查依赖包...%RESET%
python -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo %YELLOW%安装PyTorch...%RESET%
    pip install torch torchvision torchaudio
)

python -c "import websockets" >nul 2>&1
if errorlevel 1 (
    echo %YELLOW%安装websockets...%RESET%
    pip install websockets
)

python -c "import numpy" >nul 2>&1
if errorlevel 1 (
    echo %YELLOW%安装numpy...%RESET%
    pip install numpy
)

echo %GREEN%✅ 依赖检查完成%RESET%

:: 启动训练系统
echo.
echo %BLUE%🚀 启动实时训练系统...%RESET%
start "A管理模型训练系统" cmd /k "title A管理模型训练系统 && python real_training_system.py"

:: 等待服务启动
timeout /t 3 /nobreak >nul

:: 打开仪表板
echo %BLUE%📊 打开训练仪表板...%RESET%
start "" "a_manager_dashboard.html"

:: 显示状态
echo.
echo %GREEN%✅ A管理模型实时训练系统已启动！%RESET%
echo.
echo 📋 访问地址：
echo   训练仪表板: http://localhost:8766
echo   实时监控: a_manager_dashboard.html
echo.
echo 🔧 使用说明：
echo   1. 仪表板显示实时训练指标
echo   2. 可以提交测试任务观察效果
echo   3. 训练数据实时生成和更新
echo   4. 模型参数自动保存
pause