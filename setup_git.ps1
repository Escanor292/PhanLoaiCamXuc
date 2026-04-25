# Setup Git for Collaborative Training
# Run this script to initialize Git repo and push to GitHub

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "GIT SETUP - COLLABORATIVE TRAINING" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if git is installed
Write-Host "Checking Git installation..." -ForegroundColor Yellow
$gitVersion = git --version 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Git is not installed!" -ForegroundColor Red
    Write-Host "  Please install Git from: https://git-scm.com/download/win" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Git is installed: $gitVersion" -ForegroundColor Green
Write-Host ""

# Check if already a git repo
if (Test-Path ".git") {
    Write-Host "⚠ This is already a Git repository" -ForegroundColor Yellow
    $response = Read-Host "Do you want to continue? (y/n)"
    if ($response -ne "y") {
        Write-Host "Aborted." -ForegroundColor Yellow
        exit 0
    }
} else {
    # Initialize git repo
    Write-Host "Initializing Git repository..." -ForegroundColor Yellow
    git init
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Git repository initialized" -ForegroundColor Green
    } else {
        Write-Host "✗ Failed to initialize Git repository" -ForegroundColor Red
        exit 1
    }
}
Write-Host ""

# Add remote
Write-Host "Adding remote repository..." -ForegroundColor Yellow
$remoteUrl = "https://github.com/Escanor292/PhanLoaiCamXuc.git"

# Check if remote already exists
$existingRemote = git remote get-url origin 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "⚠ Remote 'origin' already exists: $existingRemote" -ForegroundColor Yellow
    $response = Read-Host "Do you want to update it? (y/n)"
    if ($response -eq "y") {
        git remote set-url origin $remoteUrl
        Write-Host "✓ Remote updated to: $remoteUrl" -ForegroundColor Green
    }
} else {
    git remote add origin $remoteUrl
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Remote added: $remoteUrl" -ForegroundColor Green
    } else {
        Write-Host "✗ Failed to add remote" -ForegroundColor Red
        exit 1
    }
}
Write-Host ""

# Setup Git LFS (optional)
Write-Host "Setting up Git LFS (for large model files)..." -ForegroundColor Yellow
$gitLfsVersion = git lfs version 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Git LFS is installed: $gitLfsVersion" -ForegroundColor Green
    
    # Install Git LFS hooks
    git lfs install
    
    # Track large files
    Write-Host "  Tracking large files..." -ForegroundColor Yellow
    git lfs track "saved_model/*.bin"
    git lfs track "saved_model/*.pt"
    git lfs track "saved_model/*.pth"
    git lfs track "model_registry/models/*/pytorch_model.bin"
    
    Write-Host "✓ Git LFS configured" -ForegroundColor Green
} else {
    Write-Host "⚠ Git LFS is not installed (optional)" -ForegroundColor Yellow
    Write-Host "  You can install it from: https://git-lfs.github.com/" -ForegroundColor Yellow
    Write-Host "  This is optional but recommended for large model files" -ForegroundColor Yellow
}
Write-Host ""

# Check .gitignore
Write-Host "Checking .gitignore..." -ForegroundColor Yellow
if (Test-Path ".gitignore") {
    Write-Host "✓ .gitignore exists" -ForegroundColor Green
} else {
    Write-Host "⚠ .gitignore not found" -ForegroundColor Yellow
    Write-Host "  Creating default .gitignore..." -ForegroundColor Yellow
    
    $gitignoreContent = @"
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.pytest_cache/

# Model files (large files)
saved_model/*.bin
saved_model/*.pt
saved_model/*.pth
model_registry/models/*/pytorch_model.bin
model_registry/backups/

# Experiments (optional)
experiments/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
"@
    
    $gitignoreContent | Out-File -FilePath ".gitignore" -Encoding UTF8
    Write-Host "✓ .gitignore created" -ForegroundColor Green
}
Write-Host ""

# Show status
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "GIT STATUS" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
git status
Write-Host ""

# Ask to commit
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "READY TO COMMIT" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Review the files above" -ForegroundColor White
Write-Host "  2. Run: git add ." -ForegroundColor White
Write-Host "  3. Run: git commit -m 'Initial commit'" -ForegroundColor White
Write-Host "  4. Run: git push -u origin main" -ForegroundColor White
Write-Host ""

$response = Read-Host "Do you want to commit and push now? (y/n)"
if ($response -eq "y") {
    Write-Host ""
    Write-Host "Adding files..." -ForegroundColor Yellow
    git add .
    
    Write-Host "Committing..." -ForegroundColor Yellow
    git commit -m "Initial commit: Multi-label Emotion Classification System

- Complete training pipeline
- Model registry with auto-upgrade
- Collaborative training workflow
- Documentation and tools"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Committed successfully" -ForegroundColor Green
        Write-Host ""
        
        Write-Host "Pushing to GitHub..." -ForegroundColor Yellow
        
        # Check if main branch exists
        $currentBranch = git branch --show-current
        if ($currentBranch -ne "main") {
            Write-Host "  Renaming branch to 'main'..." -ForegroundColor Yellow
            git branch -M main
        }
        
        git push -u origin main
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host ""
            Write-Host "======================================================================" -ForegroundColor Green
            Write-Host "✓ SUCCESS!" -ForegroundColor Green
            Write-Host "======================================================================" -ForegroundColor Green
            Write-Host ""
            Write-Host "Your code has been pushed to GitHub!" -ForegroundColor Green
            Write-Host "Repository: https://github.com/Escanor292/PhanLoaiCamXuc.git" -ForegroundColor Cyan
            Write-Host ""
            Write-Host "Team members can now clone the repo:" -ForegroundColor Yellow
            Write-Host "  git clone https://github.com/Escanor292/PhanLoaiCamXuc.git" -ForegroundColor White
            Write-Host ""
        } else {
            Write-Host ""
            Write-Host "✗ Push failed" -ForegroundColor Red
            Write-Host "  This might be because:" -ForegroundColor Yellow
            Write-Host "  1. You need to authenticate with GitHub" -ForegroundColor Yellow
            Write-Host "  2. The repository already has content" -ForegroundColor Yellow
            Write-Host ""
            Write-Host "Try pushing manually:" -ForegroundColor Yellow
            Write-Host "  git push -u origin main" -ForegroundColor White
            Write-Host ""
        }
    } else {
        Write-Host "✗ Commit failed" -ForegroundColor Red
    }
} else {
    Write-Host ""
    Write-Host "Skipped commit and push." -ForegroundColor Yellow
    Write-Host "You can do it manually later:" -ForegroundColor Yellow
    Write-Host "  git add ." -ForegroundColor White
    Write-Host "  git commit -m 'Initial commit'" -ForegroundColor White
    Write-Host "  git push -u origin main" -ForegroundColor White
    Write-Host ""
}

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "SETUP COMPLETE" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Read GITHUB_SETUP.md for detailed instructions" -ForegroundColor Yellow
Write-Host ""
