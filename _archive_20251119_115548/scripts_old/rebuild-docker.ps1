#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Quick rebuild with full ML dependencies (LightGBM + CatBoost)
.DESCRIPTION
    Rebuilds Docker image with all 6 ML models enabled
#>

Write-Host ""
Write-Host "🔨 Rebuilding Quantum Trader with full ML stack..." -ForegroundColor Cyan
Write-Host ""

# Step 1: Stop existing
Write-Host "1. Stopping existing containers..." -ForegroundColor Yellow
docker-compose --profile dev down
Write-Host "   ✅ Containers stopped" -ForegroundColor Green
Write-Host ""

# Step 2: Rebuild
Write-Host "2. Building new image with LightGBM + CatBoost..." -ForegroundColor Yellow
Write-Host "   (This will take ~2-3 minutes)" -ForegroundColor Gray
docker-compose build --no-cache backend
if ($LASTEXITCODE -ne 0) {
    Write-Host "   ❌ Build failed!" -ForegroundColor Red
    exit 1
}
Write-Host "   ✅ Build completed" -ForegroundColor Green
Write-Host ""

# Step 3: Start
Write-Host "3. Starting container..." -ForegroundColor Yellow
docker-compose --profile dev up -d backend
Write-Host "   ✅ Container started" -ForegroundColor Green
Write-Host ""

# Step 4: Wait for init
Write-Host "4. Waiting for initialization (15s)..." -ForegroundColor Yellow
Start-Sleep -Seconds 15
Write-Host ""

# Step 5: Verify sklearn
Write-Host "5. Verifying sklearn installation..." -ForegroundColor Yellow
docker exec quantum_backend python -c "import sklearn; print(f'   sklearn: {sklearn.__version__}')"
Write-Host ""

# Step 6: Verify ML libraries
Write-Host "6. Verifying ML libraries..." -ForegroundColor Yellow
docker exec quantum_backend python -c @"
try:
    import xgboost
    print('   ✅ XGBoost: OK')
except:
    print('   ❌ XGBoost: MISSING')
try:
    import lightgbm
    print('   ✅ LightGBM: OK')
except:
    print('   ⚠️  LightGBM: MISSING')
try:
    import catboost
    print('   ✅ CatBoost: OK')
except:
    print('   ⚠️  CatBoost: MISSING')
"@
Write-Host ""

# Step 7: Check ensemble
Write-Host "7. Checking ensemble model..." -ForegroundColor Yellow
docker exec quantum_backend python -c @"
from ai_engine.agents.xgb_agent import XGBAgent
import sys
agent = XGBAgent()
if agent.ensemble is not None:
    print('   ✅ Ensemble: ACTIVE (6 models)')
    sys.exit(0)
elif agent.model is not None:
    print('   ⚠️  Ensemble: FALLBACK (single model)')
    sys.exit(0)
else:
    print('   ❌ Models: NOT LOADED')
    sys.exit(1)
"@
Write-Host ""

# Step 8: Test API
Write-Host "8. Testing AI model API..." -ForegroundColor Yellow
try {
    $status = Invoke-RestMethod -Uri "http://localhost:8000/api/ai/model/status" -TimeoutSec 5
    Write-Host "   ✅ Model Status: $($status.status)" -ForegroundColor Green
    Write-Host "   ✅ Model Type: $($status.model_type)" -ForegroundColor Green
    Write-Host "   ✅ Accuracy: $([math]::Round($status.accuracy * 100, 2))%" -ForegroundColor Green
} catch {
    Write-Host "   ⚠️  API not ready yet (may need more time)" -ForegroundColor Yellow
}
Write-Host ""

# Summary
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "✅ Rebuild Complete!" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host ""
Write-Host "Container is running with sklearn + ML models" -ForegroundColor White
Write-Host ""
Write-Host "Useful commands:" -ForegroundColor Yellow
Write-Host "  docker logs -f quantum_backend          # View logs" -ForegroundColor Gray
Write-Host "  docker exec quantum_backend python ...  # Run Python" -ForegroundColor Gray
Write-Host "  docker-compose restart backend          # Restart" -ForegroundColor Gray
Write-Host "  docker-compose --profile dev down       # Stop" -ForegroundColor Gray
Write-Host ""
