# Scheduled Retraining Script for Quantum Trader AI Models
# Runs periodically to keep models fresh
# Usage: Add to Windows Task Scheduler to run every 24 hours

$ErrorActionPreference = "Stop"

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  QUANTUM TRADER - SCHEDULED AI MODEL RETRAINING" -ForegroundColor Cyan
Write-Host "================================================`n" -ForegroundColor Cyan

# Navigate to project root
cd C:\quantum_trader

# 1. Run Training
Write-Host "[1/4] 🔄 Training AI models..." -ForegroundColor Yellow
python scripts/train_xgboost_quick.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Training failed!" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Training complete`n" -ForegroundColor Green

# 2. Backup old models
Write-Host "[2/4] 💾 Backing up old models..." -ForegroundColor Yellow
$timestamp = (Get-Date -Format "yyyyMMdd_HHmmss")
$backupPath = "ai_engine\models\backup_$timestamp"
New-Item -ItemType Directory -Force -Path $backupPath | Out-Null

Copy-Item "ai_engine\models\xgb_model.pkl" "$backupPath\xgb_model.pkl" -ErrorAction SilentlyContinue
Copy-Item "ai_engine\models\scaler.pkl" "$backupPath\scaler.pkl" -ErrorAction SilentlyContinue

Write-Host "✅ Backup saved to $backupPath`n" -ForegroundColor Green

# 3. Deploy new models
Write-Host "[3/4] 🚀 Deploying new models..." -ForegroundColor Yellow
Copy-Item "models\xgboost_model.pkl" "ai_engine\models\xgb_model.pkl" -Force
Copy-Item "models\xgboost_scaler.pkl" "ai_engine\models\scaler.pkl" -Force
Copy-Item "models\xgboost_features.pkl" "ai_engine\models\xgboost_features.pkl" -Force

Write-Host "✅ Models deployed`n" -ForegroundColor Green

# 4. Restart backend to load new models
Write-Host "[4/4] ♻️  Restarting backend..." -ForegroundColor Yellow
docker restart quantum_backend_prod
Start-Sleep -Seconds 10

# Verify
$health = try { (Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 5).status } catch { "error" }
if ($health -eq "ok") {
    Write-Host "✅ Backend restarted successfully`n" -ForegroundColor Green
} else {
    Write-Host "⚠️  Backend may not be fully ready yet`n" -ForegroundColor Yellow
}

# Summary
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  ✅ RETRAINING COMPLETE!" -ForegroundColor Green
Write-Host "  - New models trained and deployed" -ForegroundColor White
Write-Host "  - Old models backed up to: $backupPath" -ForegroundColor White
Write-Host "  - Backend restarted with fresh models" -ForegroundColor White
Write-Host "================================================`n" -ForegroundColor Cyan

# Log timestamp
$logEntry = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') - Retraining completed successfully"
Add-Content -Path "ai_engine\models\retraining_log.txt" -Value $logEntry
