@echo off
echo ========================================
echo    Self Brain AGI System Deployment
echo ========================================
echo.
echo Project: Self Brain
echo Team: Silence Crow Team
echo Contact: silencecrowtom@qq.com
echo.
echo Starting deployment process...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.6 or higher
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "myenv" (
    echo Creating Python virtual environment...
    python -m venv myenv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo Activating virtual environment...
call myenv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

REM Install dependencies
echo Installing dependencies...
pip install --upgrade pip
if errorlevel 1 (
    echo ERROR: Failed to upgrade pip
    pause
    exit /b 1
)

pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies from requirements.txt
    echo Attempting to install minimal dependencies...
    pip install flask flask-socketio opencv-python pillow numpy requests torch torchvision
    if errorlevel 1 (
        echo ERROR: Failed to install minimal dependencies
        pause
        exit /b 1
    )
)

REM Initialize system directories and data
echo Initializing system directories and data...
python initialize_system.py
if errorlevel 1 (
    echo WARNING: System initialization had issues, but continuing...
)

REM Create model configurations
echo Creating model configurations...
python create_model_configs.py
if errorlevel 1 (
    echo WARNING: Model configuration creation had issues, but continuing...
)

REM Check for missing dependencies
echo Checking for missing dependencies...
python check_system.py
if errorlevel 1 (
    echo WARNING: System check found issues, but continuing...
)

REM Start the system
echo.
echo ========================================
echo    Starting Self Brain AGI System
echo ========================================
echo.
echo System will be available at:
echo - Main Interface: http://localhost:8081
echo - Training Control: http://localhost:8081/training
echo - Knowledge Import: http://localhost:8081/knowledge/import
echo - Camera Management: http://localhost:8081/camera_management
echo - API Status: http://localhost:8081/api/system/status
echo.
echo Press Ctrl+C to stop the system
echo.

REM Start the web interface directly
cd web_interface
python app.py

if errorlevel 1 (
    echo ERROR: Failed to start web interface
    echo Attempting alternative startup method...
    cd ..
    python start_system.py
    if errorlevel 1 (
        echo ERROR: All startup methods failed
        pause
        exit /b 1
    )
)

pause
