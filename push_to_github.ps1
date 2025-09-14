# 推送到GitHub仓库的PowerShell脚本
Write-Host "正在推送到GitHub仓库..." -ForegroundColor Green
Write-Host "请确保您已配置Git凭据" -ForegroundColor Yellow

# 检查网络连接
Write-Host "检查网络连接..." -ForegroundColor Cyan
try {
    $response = Invoke-WebRequest -Uri "https://github.com" -UseBasicParsing -TimeoutSec 5
    if ($response.StatusCode -eq 200) {
        Write-Host "网络连接正常" -ForegroundColor Green
    }
} catch {
    Write-Host "网络连接失败，请检查网络设置" -ForegroundColor Red
    pause
    exit 1
}

# 设置远程仓库
Write-Host "设置远程仓库..." -ForegroundColor Cyan
git remote set-url origin https://github.com/Sum-Outman/self-brain.git

# 推送更改
Write-Host "正在推送更改到main分支..." -ForegroundColor Cyan
try {
    git push -u origin main
    if ($LASTEXITCODE -eq 0) {
        Write-Host "成功推送到GitHub仓库！" -ForegroundColor Green
    } else {
        Write-Host "推送失败，请检查Git凭据和网络设置" -ForegroundColor Red
        Write-Host "您可能需要：" -ForegroundColor Yellow
        Write-Host "1. 配置GitHub个人访问令牌" -ForegroundColor Yellow
        Write-Host "2. 检查SSH密钥配置" -ForegroundColor Yellow
        Write-Host "3. 使用GitHub Desktop作为替代方案" -ForegroundColor Yellow
    }
} catch {
    Write-Host "推送过程中出现错误：$($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "按任意键退出..." -ForegroundColor Cyan
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")