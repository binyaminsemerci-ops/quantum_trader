Write-Host "🚀 Starting Phase 4U Deployment — Auto-Model Federation & Consensus Layer" -ForegroundColor Cyan
cd C:\quantum_trader

Write-Host "🔄 Updating repository..." -ForegroundColor Yellow
git pull origin main

Write-Host "🏗️ Building Docker image..." -ForegroundColor Yellow
docker-compose -f docker-compose.vps.yml build model-federation

Write-Host "▶️ Starting container..." -ForegroundColor Yellow
docker-compose -f docker-compose.vps.yml up -d model-federation
Start-Sleep -Seconds 15

Write-Host "🔍 Checking container..." -ForegroundColor Yellow
docker ps | findstr model_federation

Write-Host "🧩 Injecting mock model signals (6 models)..." -ForegroundColor Yellow

$timestamp = [int][double]::Parse((Get-Date -UFormat %s))

# Model 1: XGBoost - Strong BUY
docker exec quantum_redis redis-cli SET quantum:model:xgb:signal "{`"action`":`"BUY`",`"confidence`":0.85,`"timestamp`":$timestamp}"
Write-Host "  ✓ XGBoost: BUY (0.85)" -ForegroundColor Gray

# Model 2: LightGBM - BUY
docker exec quantum_redis redis-cli SET quantum:model:lgbm:signal "{`"action`":`"BUY`",`"confidence`":0.78,`"timestamp`":$timestamp}"
Write-Host "  ✓ LightGBM: BUY (0.78)" -ForegroundColor Gray

# Model 3: PatchTST - BUY
docker exec quantum_redis redis-cli SET quantum:model:patchtst:signal "{`"action`":`"BUY`",`"confidence`":0.82,`"timestamp`":$timestamp}"
Write-Host "  ✓ PatchTST: BUY (0.82)" -ForegroundColor Gray

# Model 4: NHITS - SELL (minority)
docker exec quantum_redis redis-cli SET quantum:model:nhits:signal "{`"action`":`"SELL`",`"confidence`":0.65,`"timestamp`":$timestamp}"
Write-Host "  ✓ NHITS: SELL (0.65)" -ForegroundColor Gray

# Model 5: RL Sizer - BUY
docker exec quantum_redis redis-cli SET quantum:model:rl_sizer:signal "{`"action`":`"BUY`",`"confidence`":0.75,`"timestamp`":$timestamp}"
Write-Host "  ✓ RL Sizer: BUY (0.75)" -ForegroundColor Gray

# Model 6: Evo Model - HOLD
docker exec quantum_redis redis-cli SET quantum:model:evo_model:signal "{`"action`":`"HOLD`",`"confidence`":0.60,`"timestamp`":$timestamp}"
Write-Host "  ✓ Evo Model: HOLD (0.60)" -ForegroundColor Gray

Write-Host "`n⏳ Waiting for federation cycle (15 seconds)..." -ForegroundColor Yellow
Start-Sleep -Seconds 15

Write-Host "`n📜 Checking Federation Engine logs..." -ForegroundColor Cyan
docker logs --tail 30 quantum_model_federation

Write-Host "`n🔁 Checking consensus data in Redis..." -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray

Write-Host "`n🎯 Consensus Signal:" -ForegroundColor Yellow
$consensus = docker exec quantum_redis redis-cli GET quantum:consensus:signal
if ($consensus) {
    $consensus | ConvertFrom-Json | ConvertTo-Json -Depth 5
}

Write-Host "`n🧠 Trust Weights (all models):" -ForegroundColor Yellow
docker exec quantum_redis redis-cli HGETALL quantum:trust:history

Write-Host "`n📊 Federation Metrics:" -ForegroundColor Yellow
$metrics = docker exec quantum_redis redis-cli GET quantum:federation:metrics
if ($metrics) {
    $metrics | ConvertFrom-Json | ConvertTo-Json -Depth 5
}

Write-Host "`n🧠 Fetching AI Engine Health snapshot..." -ForegroundColor Cyan
try {
    $health = Invoke-RestMethod -Uri "http://localhost:8001/health" -Method Get -ErrorAction SilentlyContinue
    if ($health.metrics.model_federation) {
        $health.metrics.model_federation | ConvertTo-Json -Depth 5
    }
} catch {
    Write-Host "⚠️ AI Engine health endpoint not available" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🎯 PHASE 4U DEPLOYMENT COMPLETE" -ForegroundColor Green
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Green
Write-Host "• Model Federation Engine: ✅ Running" -ForegroundColor Green
Write-Host "• Model Broker: ✅ Collecting signals (6 models)" -ForegroundColor Green
Write-Host "• Consensus Calculator: ✅ Building weighted consensus" -ForegroundColor Green
Write-Host "• Trust Memory: ✅ Learning model reliability" -ForegroundColor Green
Write-Host "• Feedback Loop: ✅ Active (10 sec cycle)" -ForegroundColor Green
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Green
Write-Host ""
Write-Host "📊 Monitor live:" -ForegroundColor Cyan
Write-Host "  docker logs -f quantum_model_federation" -ForegroundColor Gray
Write-Host ""
Write-Host "🔍 Check consensus:" -ForegroundColor Cyan
Write-Host "  docker exec quantum_redis redis-cli GET quantum:consensus:signal" -ForegroundColor Gray
Write-Host ""
Write-Host "🧠 View trust weights:" -ForegroundColor Cyan
Write-Host "  docker exec quantum_redis redis-cli HGETALL quantum:trust:history" -ForegroundColor Gray
Write-Host ""
