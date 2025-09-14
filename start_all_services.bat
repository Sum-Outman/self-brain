@echo off
echo Starting Self Brain AGI System...
echo ================================

REM 启动主系统
echo Starting Main Web Interface (5000)...
start cmd /k "python web_interface/working_enhanced_chat.py"
timeout /t 2 /nobreak > nul

echo Starting A Management Model (5001)...
start cmd /k "python a_management_server.py"
timeout /t 2 /nobreak > nul

echo Starting B Language Model (5002)...
start cmd /k "python sub_models/B_language/app.py"
timeout /t 2 /nobreak > nul

echo Starting C Audio Model (5003)...
start cmd /k "python sub_models/C_audio/api.py"
timeout /t 2 /nobreak > nul

echo Starting D Image Model (5004)...
start cmd /k "python sub_models/D_image/api.py"
timeout /t 2 /nobreak > nul

echo Starting E Video Model (5005)...
start cmd /k "python sub_models/E_video/api.py"
timeout /t 2 /nobreak > nul

echo Starting F Spatial Model (5006)...
start cmd /k "python sub_models/F_spatial/api.py"
timeout /t 2 /nobreak > nul

echo Starting G Sensor Model (5007)...
start cmd /k "python sub_models/G_sensor/api.py"
timeout /t 2 /nobreak > nul

echo Starting H Computer Control (5008)...
start cmd /k "python sub_models/H_computer_control/api.py"
timeout /t 2 /nobreak > nul

echo Starting I Knowledge Model (5009)...
start cmd /k "python sub_models/I_knowledge/api.py"
timeout /t 2 /nobreak > nul

echo Starting J Motion Model (5010)...
start cmd /k "python sub_models/J_motion/api.py"
timeout /t 2 /nobreak > nul

echo Starting K Programming Model (5011)...
start cmd /k "python sub_models/K_programming/programming_api.py"
timeout /t 2 /nobreak > nul

echo Starting Training Manager (5012)...
start cmd /k "python training_manager.py"
timeout /t 2 /nobreak > nul

echo Starting Quantum Integration (5013)...
start cmd /k "python quantum_integration.py"
timeout /t 2 /nobreak > nul

echo Starting Standalone A Manager (5014)...
start cmd /k "python a_manager_standalone.py"
timeout /t 2 /nobreak > nul

echo Starting Manager Model API (5015)...
start cmd /k "python manager_model/app.py"
timeout /t 2 /nobreak > nul

echo.
echo All services are starting...
echo Please wait 30 seconds for all services to initialize...
echo.
echo Access URLs:
echo   Main Interface: http://localhost:5000
echo   Management: http://localhost:5015
echo.
pause