@echo off
python app.py > web_interface_log.txt 2>&1
echo Exit code: %errorlevel%