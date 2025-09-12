# PowerShell脚本：推送Self Brain到GitHub

Write-Host "🚀 正在推送 Self Brain 项目到 GitHub..." -ForegroundColor Green
Write-Host ""
Write-Host "📋 请确保您已经在 GitHub 上创建了仓库" -ForegroundColor Yellow
Write-Host "🔗 仓库地址格式: https://github.com/YOUR_USERNAME/self-brain.git" -ForegroundColor Cyan
Write-Host ""

$repoUrl = Read-Host "请输入您的GitHub仓库URL"

if ([string]::IsNullOrWhiteSpace($repoUrl)) {
    Write-Host "❌ 错误: 请输入有效的仓库URL" -ForegroundColor Red
    Read-Host "按任意键退出..."
    exit 1
}

Write-Host ""
Write-Host "🔗 正在添加远程仓库..." -ForegroundColor Yellow
git remote add origin $repoUrl

Write-Host ""
Write-Host "📤 正在推送到GitHub..." -ForegroundColor Yellow
git branch -M main
git push -u origin main

Write-Host ""
Write-Host "✅ 推送完成！" -ForegroundColor Green
Write-Host "🎯 项目已成功上传到GitHub" -ForegroundColor Green
Write-Host "🔗 访问地址: $repoUrl" -ForegroundColor Cyan
Read-Host "按任意键继续..."