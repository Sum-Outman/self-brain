@echo off
echo 正在推送到GitHub仓库...
echo 请确保您已配置Git凭据
pause

git push -u origin main

echo 推送完成！
pause