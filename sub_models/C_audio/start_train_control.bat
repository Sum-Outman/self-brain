@echo off
chcp 65001 > nul
echo ========================================
echo 增强版音频处理模型训练控制服务启动脚本
echo Enhanced Audio Processing Model Training Control Service Startup Script
echo ========================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python，请先安装Python 3.6或更高版本
    echo Error: Python not found, please install Python 3.6 or higher
    pause
    exit /b 1
)

REM 检查依赖是否安装
echo 检查并安装所需依赖包...
echo Checking and installing required dependencies...
pip install flask flask_cors torch torchaudio librosa soundfile numpy scipy transformers psutil

REM 创建必要的目录
echo 创建必要的目录结构...
echo Creating necessary directory structure...
if not exist models mkdir models
if not exist config mkdir config
if not exist output mkdir output
if not exist temp mkdir temp

REM 启动训练控制服务
echo 启动增强版音频训练控制服务...
echo Starting Enhanced Audio Training Control Service...
echo 服务将在 http://localhost:5006 运行
echo Service will run at http://localhost:5006
echo API文档: http://localhost:5006/api/training/status
echo API Documentation: http://localhost:5006/api/training/status
echo.

python enhanced_train_control.py

pause
