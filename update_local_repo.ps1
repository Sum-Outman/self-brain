Write-Host "Starting Self Brain AGI local repository update..." -ForegroundColor Green
Write-Host ""

# Add all changes
Write-Host "Adding all changes..." -ForegroundColor Yellow
git add .

# Commit changes
Write-Host "Committing changes..." -ForegroundColor Yellow
git commit -m "Update: Self Brain AGI system with enhanced features and documentation"

# Show commit summary
Write-Host "" -ForegroundColor Yellow
git log -1 --stat

Write-Host "" -ForegroundColor Green
Write-Host "Local repository update completed successfully!" -ForegroundColor Green
Write-Host "" -ForegroundColor Yellow
Write-Host "When network connection to GitHub is restored, please run the following command to push changes:"
Write-Host "cd d:\shiyan"
Write-Host "git push origin main"
Write-Host "" -ForegroundColor Green
Write-Host "Alternatively, you can run push_to_github.ps1 script when network is available."
Write-Host "" -ForegroundColor White
Read-Host "Press Enter to exit..."