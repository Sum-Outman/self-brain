@echo off
echo Starting A Management Model API Server...
echo.
echo Server will be available at:
echo   - http://localhost:5014
echo   - http://127.0.0.1:5014
echo.
echo Available endpoints:
echo   - GET  /api/health         - Health check
echo   - GET  /api/models         - List all models
echo   - POST /process_message    - Process messages
echo   - POST /api/emotion/analyze - Emotion analysis
echo   - GET  /api/system/stats   - System statistics
echo.

python a_manager_standalone.py
pause