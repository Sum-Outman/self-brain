@echo off
echo ================================================
echo Self Brain AGI System - Port Configuration
echo ================================================
echo.

REM 设置环境变量以避免端口冲突
set PORT_A_MANAGER=5001
set PORT_STANDALONE=5014
set PORT_MANAGER_API=5015
set PORT_K_PROGRAMMING=5011
set PORT_QUANTUM=5013
set PORT_TRAINING=5012

REM 启动各个服务
echo Starting A Management Model Service (Port: %PORT_A_MANAGER%)...
start cmd /k "cd /d %~dp0 && python a_management_server.py --port=%PORT_A_MANAGER%"

echo Starting Standalone A Manager (Port: %PORT_STANDALONE%)...
start cmd /k "cd /d %~dp0 && python a_manager_standalone.py --port=%PORT_STANDALONE%"

echo Starting Manager Model API (Port: %PORT_MANAGER_API%)...
start cmd /k "cd /d %~dp0manager_model && python app.py --port=%PORT_MANAGER_API%"

echo Starting K Programming Model (Port: %PORT_K_PROGRAMMING%)...
start cmd /k "cd /d %~dp0sub_models\K_programming && python programming_api.py --port=%PORT_K_PROGRAMMING%"

echo Starting Quantum Integration (Port: %PORT_QUANTUM%)...
start cmd /k "cd /d %~dp0 && python sub_models\quantum_integration.py --port=%PORT_QUANTUM%"

echo.
echo ================================================
echo All services starting...
echo ================================================
echo.
echo Service URLs:
echo   - A Management Model:  http://localhost:%PORT_A_MANAGER%/api/health
echo   - Standalone A Manager: http://localhost:%PORT_STANDALONE%/api/health
echo   - Manager Model API: http://localhost:%PORT_MANAGER_API%/api/health
echo   - K Programming: http://localhost:%PORT_K_PROGRAMMING%/api/health
echo   - Quantum Integration: http://localhost:%PORT_QUANTUM%/api/health
echo.
echo Press any key to continue...
pause >nul