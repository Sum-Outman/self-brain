# Self Brain AGI System Deployment Script (PowerShell)
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   Self Brain AGI System Deployment" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Project: Self Brain" -ForegroundColor Yellow
Write-Host "Team: Silence Crow Team" -ForegroundColor Yellow
Write-Host "Contact: silencecrowtom@qq.com" -ForegroundColor Yellow
Write-Host ""
Write-Host "Starting deployment process..." -ForegroundColor Green

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Python not found"
    }
    Write-Host "Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.6 or higher" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Create virtual environment if it doesn't exist
if (!(Test-Path "myenv")) {
    Write-Host "Creating Python virtual environment..." -ForegroundColor Yellow
    python -m venv myenv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to create virtual environment" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\myenv\Scripts\Activate.ps1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to activate virtual environment" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to upgrade pip" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install dependencies from requirements.txt" -ForegroundColor Red
    Write-Host "Attempting to install minimal dependencies..." -ForegroundColor Yellow
    python -m pip install flask flask-socketio opencv-python pillow numpy requests torch torchvision
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to install minimal dependencies" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# Initialize system directories and data
Write-Host "Initializing system directories and data..." -ForegroundColor Yellow
python initialize_system.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: System initialization had issues, but continuing..." -ForegroundColor Yellow
}

# Create model configurations
Write-Host "Creating model configurations..." -ForegroundColor Yellow
python create_model_configs.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: Model configuration creation had issues, but continuing..." -ForegroundColor Yellow
}

# Check for missing dependencies
Write-Host "Checking for missing dependencies..." -ForegroundColor Yellow
python check_system.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: System check found issues, but continuing..." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   Starting Self Brain AGI System" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "System will be available at:" -ForegroundColor Green
Write-Host "- Main Interface: http://localhost:8081" -ForegroundColor White
Write-Host "- Training Control: http://localhost:8081/training" -ForegroundColor White
Write-Host "- Knowledge Import: http://localhost:8081/knowledge/import" -ForegroundColor White
Write-Host "- Camera Management: http://localhost:8081/camera_management" -ForegroundColor White
Write-Host "- API Status: http://localhost:8081/api/system/status" -ForegroundColor White
Write-Host ""
Write-Host "Press Ctrl+C to stop the system" -ForegroundColor Yellow
Write-Host ""

# Start the web interface directly
Set-Location "web_interface"
python app.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to start web interface" -ForegroundColor Red
    Write-Host "Attempting alternative startup method..." -ForegroundColor Yellow
    Set-Location ".."
    python start_system.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: All startup methods failed" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}
