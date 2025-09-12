@echo off
echo 🎯 启动A管理模型系统
echo =================================

:: 检查Python环境
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python未安装或未添加到PATH
    pause
    exit /b 1
)

:: 安装依赖
echo 📦 检查依赖...
python -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo ⚠️ 安装PyTorch...
    pip install torch torchvision torchaudio
)

python -c "import websockets" >nul 2>&1
if errorlevel 1 (
    echo ⚠️ 安装websockets...
    pip install websockets
)

:: 启动A管理模型Web服务器
start cmd /k "cd /d %~dp0web_interface && python a_manager_server.py"
timeout /t 2

:: 启动Web界面
echo 🌐 启动Web界面...
start http://localhost:8765

:: 启动训练（可选）
echo 🚀 如需训练A管理模型，请运行: python train_a_manager.py

echo ✅ A管理模型系统启动完成！
echo 📱 请在浏览器中访问: http://localhost:8765
pause