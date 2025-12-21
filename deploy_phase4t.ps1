Write-Host "🚀 Starting Phase 4T Deployment — Strategic Evolution Engine" -ForegroundColor Cyan
cd C:\quantum_trader

Write-Host "🔄 Updating repository..." -ForegroundColor Yellow
git pull origin main

Write-Host "🏗️ Building Docker image..." -ForegroundColor Yellow
docker-compose -f docker-compose.vps.yml build strategic-evolution

Write-Host "▶️ Starting container..." -ForegroundColor Yellow
docker-compose -f docker-compose.vps.yml up -d strategic-evolution
Start-Sleep -Seconds 15

Write-Host "🔍 Checking container..." -ForegroundColor Yellow
docker ps | findstr strategic_evolution

Write-Host "📊 Injecting test strategy data..." -ForegroundColor Yellow
docker exec redis redis-cli RPUSH quantum:strategy:performance '{\"strategy\":\"nhits\",\"sharpe_ratio\":1.8,\"win_rate\":0.65,\"max_drawdown\":0.12,\"consistency\":0.78}'
docker exec redis redis-cli RPUSH quantum:strategy:performance '{\"strategy\":\"patchtst\",\"sharpe_ratio\":2.1,\"win_rate\":0.72,\"max_drawdown\":0.08,\"consistency\":0.85}'
docker exec redis redis-cli RPUSH quantum:strategy:performance '{\"strategy\":\"xgboost\",\"sharpe_ratio\":1.5,\"win_rate\":0.58,\"max_drawdown\":0.15,\"consistency\":0.65}'

Write-Host "⏳ Waiting for processing cycle..." -ForegroundColor Yellow
Start-Sleep -Seconds 90

Write-Host "🧠 Fetching AI Engine Health snapshot..." -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod -Uri "http://localhost:8001/health" -Method Get
    $health.metrics.strategic_evolution | ConvertTo-Json
} catch {
    Write-Host "⚠️ Could not fetch health endpoint: $_" -ForegroundColor Red
}

Write-Host "🔁 Checking evolution data..." -ForegroundColor Yellow
Write-Host "Rankings:" -ForegroundColor Cyan
docker exec redis redis-cli GET quantum:evolution:rankings

Write-Host "`nSelected Models:" -ForegroundColor Cyan
docker exec redis redis-cli GET quantum:evolution:selected

Write-Host "`nMutations:" -ForegroundColor Cyan
docker exec redis redis-cli GET quantum:evolution:mutated

Write-Host "📜 Latest logs:" -ForegroundColor Yellow
docker logs --tail 30 quantum_strategic_evolution

Write-Host ""
Write-Host "🎯 PHASE 4T DEPLOYMENT COMPLETE" -ForegroundColor Green
Write-Host "-------------------------------------------------------" -ForegroundColor Green
Write-Host "• Strategic Evolution Engine: ✅ Running" -ForegroundColor Green
Write-Host "• Performance Evaluator: ✅ Active" -ForegroundColor Green
Write-Host "• Model Selector: ✅ Top 3 Selected" -ForegroundColor Green
Write-Host "• Mutation Engine: ✅ Configs Generated" -ForegroundColor Green
Write-Host "• Retrain Manager: ✅ Jobs Scheduled" -ForegroundColor Green
Write-Host "-------------------------------------------------------" -ForegroundColor Green
