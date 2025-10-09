@echo off

REM 启动完整的管理模型服务 (manager_model/app.py) 使用端口5015
start cmd /k "cd /d d:\shiyan\manager_model && echo Starting Complete A Management Model (Port 5015)... && python app.py"

echo Waiting for Complete A Management Model to start...
timeout /t 5 /nobreak

REM 启动独立简化版的A管理模型 (a_manager_standalone.py) 使用端口5014
start cmd /k "cd /d d:\shiyan && echo Starting Standalone A Management Model (Port 5014)... && python a_manager_standalone.py"

echo Waiting for Standalone A Management Model to start...
timeout /t 5 /nobreak

REM 测试两个服务是否正常运行
echo Testing Complete A Management Model (Port 5015)...
curl http://localhost:5015/api/health

echo.
echo Testing Standalone A Management Model (Port 5014)...
curl http://localhost:5014/api/health

pause