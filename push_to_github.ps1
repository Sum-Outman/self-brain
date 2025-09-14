# Self Brain AGI - GitHub Push Script
Write-Host "Starting Self Brain AGI GitHub push process..." -ForegroundColor Green
Write-Host ""

# 检查网络连接
Write-Host "Testing GitHub connectivity..." -ForegroundColor Yellow
$canConnect = Test-Connection github.com -Count 1 -Quiet

if (-not $canConnect) {
    Write-Host "Network connection failed. Please check your internet connection." -ForegroundColor Red
    Read-Host "Press Enter to exit..."
    exit 1
}

Write-Host "Network connection OK." -ForegroundColor Green
Write-Host ""

# 设置Git配置
Write-Host "Configuring Git..." -ForegroundColor Yellow
git config --global user.name "Sum-Outman"
git config --global user.email "sum.outman@gmail.com"

# 添加所有更改
Write-Host "Adding all changes..." -ForegroundColor Yellow
git add .

# 提交更改
Write-Host "Committing changes..." -ForegroundColor Yellow
git commit -m "feat: Update Self Brain AGI system with enhanced features and documentation

- Add comprehensive documentation and usage guides
- Implement enhanced management system with A-K model integration
- Add new web interface features and knowledge management
- Update help system with simplified English content
- Add system monitoring and performance optimization
- Include training system improvements
- Add API documentation and installation guides
- Fix various system stability issues"

# 尝试HTTPS推送
Write-Host "Attempting HTTPS push..." -ForegroundColor Yellow
git remote set-url origin https://github.com/Sum-Outman/self-brain.git

# 强制推送（覆盖远程仓库）
git push origin main --force

if ($LASTEXITCODE -ne 0) {
    Write-Host "HTTPS push failed, trying SSH..." -ForegroundColor Red
    git remote set-url origin git@github.com:Sum-Outman/self-brain.git
    git push origin main --force
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Both HTTPS and SSH push failed." -ForegroundColor Red
        Write-Host "Please check the following:" -ForegroundColor Yellow
        Write-Host "1. GitHub Personal Access Token (for HTTPS)" -ForegroundColor White
        Write-Host "2. SSH key configuration (for SSH)" -ForegroundColor White
        Write-Host "3. GitHub repository permissions" -ForegroundColor White
        Write-Host "4. Network firewall settings" -ForegroundColor White
        Write-Host ""
        Write-Host "Alternative: Use GitHub Desktop to push manually" -ForegroundColor Cyan
    } else {
        Write-Host "Successfully pushed via SSH!" -ForegroundColor Green
    }
} else {
    Write-Host "Successfully pushed via HTTPS!" -ForegroundColor Green
}

Write-Host ""
Write-Host "Push process completed." -ForegroundColor Green
Read-Host "Press Enter to exit..."