@echo off
echo Starting Intelligent Management System...
echo 注意：此版本已优化启动流程

REM 检查虚拟环境是否存在
if not exist "myenv\Scripts\activate.bat" (
    echo 错误：未找到Python虚拟环境
    echo 请先创建虚拟环境: python -m venv myenv
    pause
    exit /b 1
)

REM 激活Python虚拟环境并启动系统
call myenv\Scripts\activate.bat
if errorlevel 1 (
    echo 错误：激活虚拟环境失败
    pause
    exit /b 1
)

python start_system.py
if errorlevel 1 (
    echo 错误：启动系统失败
    pause
    exit /b 1
)

echo 系统已成功启动
pause
