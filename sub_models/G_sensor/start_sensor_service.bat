@echo off
REM 传感器集成服务启动脚本
REM Sensor Integration Service Startup Script

echo 正在启动传感器集成服务...
echo Starting Sensor Integration Service...

REM 检查Python是否安装
python --version >nul 2>nul
if %errorlevel% neq 0 (
    echo 错误: 未找到Python。请先安装Python。
    echo Error: Python not found. Please install Python first.
    pause
    exit /b 1
)

REM 检查依赖是否已安装
pip show torch opencv-python redis pyaudio pyserial flask flask-cors requests numpy >nul 2>nul
if %errorlevel% neq 0 (
    echo 正在安装必要的依赖包...
    echo Installing required dependencies...
    pip install torch opencv-python redis pyaudio pyserial flask flask-cors requests numpy
    if %errorlevel% neq 0 (
        echo 错误: 依赖安装失败。
        echo Error: Failed to install dependencies.
        pause
        exit /b 1
    )
)

REM 设置环境变量
set PYTHONPATH=%PYTHONPATH%;%~dp0..\..

REM 启动传感器服务
python "%~dp0sensor_integration.py"

if %errorlevel% neq 0 (
    echo 错误: 传感器服务启动失败。
    echo Error: Failed to start sensor service.
    pause
    exit /b 1
)

pause