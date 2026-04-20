# Auto Training and Deployment Script
# Tự động merge data, training, và deploy nếu model tốt hơn

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "AUTO TRAINING AND DEPLOYMENT" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

# Bước 1: Pull code mới nhất
Write-Host "[1/5] Pulling latest code..." -ForegroundColor Yellow
git pull
Write-Host ""

# Bước 2: Merge data
Write-Host "[2/5] Merging data..." -ForegroundColor Yellow
python merge_data.py data/member_*.csv --output data/master_dataset_vi.csv --conflict-strategy merge
Write-Host ""

# Bước 3: Bật AUTO_DEPLOY
Write-Host "[3/5] Enabling AUTO_DEPLOY..." -ForegroundColor Yellow
$env:AUTO_DEPLOY = "true"
Write-Host "✓ AUTO_DEPLOY = true" -ForegroundColor Green
Write-Host ""

# Bước 4: Training
Write-Host "[4/5] Training model..." -ForegroundColor Yellow
python train_with_args.py `
    --data data/master_dataset_vi.csv `
    --epochs 5 `
    --register-model `
    --experiment-name "Auto-training $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
Write-Host ""

# Bước 5: Commit registry
Write-Host "[5/5] Committing results..." -ForegroundColor Yellow
git add model_registry/registry.json
git commit -m "Auto-training: $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
git push
Write-Host ""

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "COMPLETE!" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To check deployed model:" -ForegroundColor Yellow
Write-Host "  python model_registry.py list" -ForegroundColor White
Write-Host ""
Write-Host "To test prediction:" -ForegroundColor Yellow
Write-Host "  python my_test.py" -ForegroundColor White
Write-Host ""
