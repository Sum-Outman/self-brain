# 增强版音频处理模型训练控制服务启动脚本 (PowerShell版本)
# Enhanced Audio Processing Model Training Control Service Startup Script (PowerShell Version)

Write-Host "========================================" -ForegroundColor Green
Write-Host "增强版音频处理模型训练控制服务启动脚本" -ForegroundColor Yellow
Write-Host "Enhanced Audio Processing Model Training Control Service Startup Script" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# 检查Python是否安装
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python版本: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "错误: 未找到Python，请先安装Python 3.6或更高版本" -ForegroundColor Red
    Write-Host "Error: Python not found, please install Python 3.6 or higher" -ForegroundColor Red
    pause
    exit 1
}

# 检查并安装依赖
Write-Host "检查并安装所需依赖包..." -ForegroundColor Yellow
Write-Host "Checking and installing required dependencies..." -ForegroundColor Yellow

$packages = @("flask", "flask_cors", "torch", "torchaudio", "librosa", "soundfile", "numpy", "scipy", "transformers", "psutil")

foreach ($package in $packages) {
    try {
        pip install $package
        Write-Host "已安装/更新: $package" -ForegroundColor Green
    } catch {
        Write-Host "安装 $package 时出错" -ForegroundColor Red
    }
}

# 创建必要的目录
Write-Host "创建必要的目录结构..." -ForegroundColor Yellow
Write-Host "Creating necessary directory structure..." -ForegroundColor Yellow

$directories = @("models", "config", "output", "temp")
foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
        Write-Host "创建目录: $dir" -ForegroundColor Green
    }
}

# 启动训练控制服务
Write-Host "启动增强版音频训练控制服务..." -ForegroundColor Yellow
Write-Host "Starting Enhanced Audio Training Control Service..." -ForegroundColor Yellow
Write-Host "服务将在 http://localhost:5006 运行" -ForegroundColor Cyan
Write-Host "Service will run at http://localhost:5006" -ForegroundColor Cyan
Write-Host "API文档: http://localhost:5006/api/training/status" -ForegroundColor Cyan
Write-Host "API Documentation: http://localhost:5006/api/training/status" -ForegroundColor Cyan
Write-Host ""

# 启动Python服务
python enhanced_train_control.py

# 如果服务意外退出，暂停以便查看错误信息
Write-Host "服务已停止，按任意键退出..." -ForegroundColor Red
Write-Host "Service stopped, press any key to exit..." -ForegroundColor Red
pause
