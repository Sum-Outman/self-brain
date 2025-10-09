@echo off
setlocal enabledelayedexpansion

echo Self Brain AGI - Publishing to GitHub Repository
 echo ============================================

echo Checking if git is available...
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Git is not installed or not in PATH
    pause
    exit /b 1
)

echo.
echo Setting up Git configuration...
git config user.name "Self Brain Team"
git config user.email "silencecrowtom@qq.com"

echo.
echo Checking GitHub connectivity...
ping -n 1 github.com >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Cannot connect to GitHub. Please check your internet connection.
    pause
    exit /b 1
)

echo.
echo Configuring remote repository...
git remote set-url origin https://github.com/Sum-Outman/self-brain.git >nul 2>&1
if %errorlevel% neq 0 (
    echo Setting up new remote repository...
    git remote add origin https://github.com/Sum-Outman/self-brain.git
)

echo.
echo Checking if .gitignore exists...
if not exist .gitignore (
    echo Creating .gitignore file...
    echo # Python
    echo __pycache__/ >> .gitignore
    echo *.py[cod] >> .gitignore
    echo *$py.class >> .gitignore
    echo 
    echo # Virtual environments
    echo venv/ >> .gitignore
    echo env/ >> .gitignore
    echo .env >> .gitignore
    echo 
    echo # Logs
    echo logs/ >> .gitignore
    echo *.log >> .gitignore
    echo 
    echo # Runtime data
    echo pids >> .gitignore
    echo *.pid >> .gitignore
    echo *.seed >> .gitignore
    echo *.pid.lock >> .gitignore
    echo 
    echo # Coverage directory used by tools like istanbul
    echo coverage/ >> .gitignore
    echo *.lcov >> .gitignore
    echo 
    echo # Dependency directories
    echo node_modules/ >> .gitignore
    echo jspm_packages/ >> .gitignore
    echo 
    echo # Optional npm cache directory
    echo .npm >> .gitignore
    echo 
    echo # Optional eslint cache
    echo .eslintcache >> .gitignore
    echo 
    echo # Microbundle cache
    echo .rpt2_cache/ >> .gitignore
    echo .rts2_cache_cjs/ >> .gitignore
    echo .rts2_cache_es/ >> .gitignore
    echo .rts2_cache_umd/ >> .gitignore
    echo 
    echo # Optional REPL history
    echo .node_repl_history >> .gitignore
    echo 
    echo # Output of 'npm pack'
    echo *.tgz >> .gitignore
    echo 
    echo # Yarn Integrity file
    echo .yarn-integrity >> .gitignore
    echo 
    echo # dotenv environment variable files
    echo .env.development.local >> .gitignore
    echo .env.test.local >> .gitignore
    echo .env.production.local >> .gitignore
    echo .env.local >> .gitignore
    echo 
    echo # parcel-bundler cache (https://parceljs.org/)
    echo .cache >> .gitignore
    echo .parcel-cache >> .gitignore
    echo 
    echo # Next.js build output
    echo .next >> .gitignore
    echo out >> .gitignore
    echo 
    echo # Nuxt.js build / generate output
    echo .nuxt >> .gitignore
    echo dist >> .gitignore
    echo 
    echo # Storybook build outputs
    echo .out >> .gitignore
    echo .storybook-out >> .gitignore
    echo 
    echo # Temporary folders
    echo tmp/ >> .gitignore
    echo temp/ >> .gitignore
    echo 
    echo # OS generated files
    echo Thumbs.db >> .gitignore
    echo .DS_Store >> .gitignore
    echo .DS_Store? >> .gitignore
    echo ._* >> .gitignore
    echo .Spotlight-V100 >> .gitignore
    echo .Trashes >> .gitignore
    echo ehthumbs.db >> .gitignore
    echo desktop.ini >> .gitignore
    echo 
    echo # Editor directories and files
    echo .idea/ >> .gitignore
    echo .vscode/ >> .gitignore
    echo *.suo >> .gitignore
    echo *.ntvs* >> .gitignore
    echo *.njsproj >> .gitignore
    echo *.sln >> .gitignore
    echo *.sw? >> .gitignore
)

echo.
echo Adding all files to Git...
git add .
if %errorlevel% neq 0 (
    echo Error: Failed to add files. Check for permission issues.
    pause
    exit /b 1
)

echo.
echo Committing changes...
git commit -m "Initial commit: Self Brain AGI System - Complete codebase"
if %errorlevel% neq 0 (
    echo Warning: Commit failed. Trying with force commit...
    git commit -m "Initial commit: Self Brain AGI System - Complete codebase" --allow-empty
    if %errorlevel% neq 0 (
        echo Error: Failed to commit changes.
        pause
        exit /b 1
    )
)

echo.
echo Setting main branch...
git branch -M main

echo.
echo Pushing to GitHub repository...
git push -u origin main

if %errorlevel% neq 0 (
    echo.
    echo Error: Push failed. Trying SSH method as fallback...
    git remote set-url origin git@github.com:Sum-Outman/self-brain.git
    git push -u origin main
    
    if %errorlevel% neq 0 (
        echo.
        echo Error: Both HTTPS and SSH push methods failed.
        echo Possible reasons:
        echo 1. Invalid credentials
        echo 2. Insufficient permissions
        echo 3. Network issues
        echo 4. Repository not properly set up
        echo.
        echo Please try manual push using GitHub Desktop or verify your GitHub account settings.
    ) else (
        echo.
        echo Success: Files uploaded to GitHub using SSH method!
    )
) else (
    echo.
    echo Success: All files published to GitHub repository!
    echo Repository URL: https://github.com/Sum-Outman/self-brain
)

pause
endlocal