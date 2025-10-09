@echo off
echo ================================================
echo Self Brain AGI System - Port Configuration
echo ================================================
echo.

REM 设置环境变量以避免端口冲突
set PORT_A_MANAGER=5000
set PORT_AGI_CORE=5014
set PORT_MANAGER_API=5015
set PORT_K_PROGRAMMING=5010
set PORT_TRAINING=5012

REM 启动各个服务
echo Starting A Management Model Service (Port: %PORT_A_MANAGER%)...
start cmd /k "cd /d %~dp0 && python a_management_server.py --port=%PORT_A_MANAGER%"

echo Starting AGI Core (Port: %PORT_AGI_CORE%)...
start cmd /k "cd /d %~dp0 && python a_manager_standalone.py --port=%PORT_AGI_CORE%"

echo Starting Manager Model API (Port: %PORT_MANAGER_API%)...
start cmd /k "cd /d %~dp0manager_model && python app.py --port=%PORT_MANAGER_API%"

echo Starting K Programming Model (Port: %PORT_K_PROGRAMMING%)...
start cmd /k "cd /d %~dp0sub_models\K_programming && python programming_api.py --port=%PORT_K_PROGRAMMING%"

echo Starting Training Manager (Port: %PORT_TRAINING%)...
start cmd /k "cd /d %~dp0training_manager && python app.py --port=%PORT_TRAINING%"

echo.
echo =================================================
echo All services starting...
echo =================================================
echo.
echo Service URLs:
echo   - A Management Model:  http://localhost:%PORT_A_MANAGER%/api/health
echo   - AGI Core: http://localhost:%PORT_AGI_CORE%/api/health
echo   - Manager Model API: http://localhost:%PORT_MANAGER_API%/api/health
echo   - K Programming: http://localhost:%PORT_K_PROGRAMMING%/api/health
echo   - Training Manager: http://localhost:%PORT_TRAINING%/api/health
echo.
echo Press any key to continue...
pause >nul