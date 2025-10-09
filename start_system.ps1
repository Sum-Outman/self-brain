
# Integrated Intelligent Management System Startup Script
Write-Host "Starting Manager Service (port 5000)..." -ForegroundColor Cyan
Start-Job -ScriptBlock { python "manager_model/task_scheduler.py" }

# 启动训练管理器
Write-Host "Starting Training Manager (port 5010)..." -ForegroundColor Cyan
Start-Job -ScriptBlock { python "training_manager/train_scheduler.py" }

Write-Host "Starting Knowledge Model Service (port 5009)..." -ForegroundColor Cyan
Start-Job -ScriptBlock { python "sub_models/J_knowledge/app.py" }

Write-Host "Starting Computer Control Service (port 5007)..." -ForegroundColor Cyan
Start-Job -ScriptBlock { python "sub_models/H_computer_control/app.py" }

Write-Host "Starting Quantum Integration Service (port 5011)..." -ForegroundColor Cyan
Start-Job -ScriptBlock { python "sub_models/quantum_integration.py" }

Write-Host "Starting Web Interface (port 8080)..." -ForegroundColor Cyan
$frontendPath = Join-Path $PSScriptRoot "web_interface\frontend"
Start-Job -ScriptBlock {
    python -m http.server 8080 --directory $using:frontendPath
}

Write-Host "All services started!" -ForegroundColor Green
Write-Host "Access Web Interface: http://localhost:8080" -ForegroundColor Yellow
Write-Host "Manager API Docs: http://localhost:5000" -ForegroundColor Yellow
Write-Host "Training Manager: http://localhost:5010" -ForegroundColor Yellow
Write-Host "Quantum API: http://localhost:5011" -ForegroundColor Yellow

Write-Host "Press any key to exit..." -ForegroundColor Gray
[void][System.Console]::ReadKey($true)
