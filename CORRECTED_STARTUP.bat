@echo off
echo =================================
echo Self Brain AGI System - Port Configuration Fixed
echo =================================
echo.
echo Standard Port Configuration:
echo   5000 - Main Web Interface
echo   5001 - A Management Model
echo   5002 - B Language Model
echo   5003 - C Audio Model
echo   5004 - D Image Model
echo   5005 - E Video Model
echo   5006 - F Spatial Model
echo   5007 - G Sensor Model
echo   5008 - H Computer Control
echo   5009 - I Knowledge Model
echo   5010 - J Motion Model
echo   5011 - K Programming Model
echo   5012 - Training Manager
echo   5013 - Quantum Integration
echo   5014 - Standalone A Manager
echo   5015 - Manager Model API
echo.
echo Starting system...

REM 启动核心服务
echo 启动主界面 (5000)...
start cmd /k "cd /d d:\shiyan && python web_interface/working_enhanced_chat.py"
timeout /t 3 /nobreak > nul

echo 启动管理模型 (5001)...
start cmd /k "cd /d d:\shiyan && python a_management_server.py"
timeout /t 3 /nobreak > nul

echo 启动管理API (5015)...
start cmd /k "cd /d d:\shiyan && python manager_model/app.py"
timeout /t 3 /nobreak > nul

echo 启动训练管理器 (5012)...
start cmd /k "cd /d d:\shiyan && python training_manager.py"
timeout /t 3 /nobreak > nul

echo 启动量子集成 (5013)...
start cmd /k "cd /d d:\shiyan && python quantum_integration.py"
timeout /t 3 /nobreak > nul

echo.
echo Core services started successfully!
echo.
echo Access URLs:
echo   Main Interface: http://localhost:5000
echo   Management Interface: http://localhost:5015
echo.
echo To start remaining services, run: start_all_services.bat
echo.
pause