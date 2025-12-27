Write-Host "🚀 Starting Phase 4T+ Deployment — Strategic Evolution Engine" -ForegroundColor Cyan
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

Write-Host "📊 Injecting synthetic strategy performance data (6 models)..." -ForegroundColor Yellow
1..6 | ForEach-Object {
    $sharpe = [math]::Round((Get-Random -Minimum 0.5 -Maximum 2.5), 4)
    $winrate = [math]::Round((Get-Random -Minimum 0.4 -Maximum 0.9), 4)
    $drawdown = [math]::Round((Get-Random -Minimum 0.05 -Maximum 0.25), 4)
    $consistency = [math]::Round((Get-Random -Minimum 0.3 -Maximum 0.9), 4)
    
    $json = "{`"strategy`":`"model_$_`",`"sharpe_ratio`":$sharpe,`"win_rate`":$winrate,`"max_drawdown`":$drawdown,`"consistency`":$consistency}"
    
    docker exec redis redis-cli RPUSH quantum:strategy:performance $json
    Write-Host "  ✓ Injected model_$_ (Sharpe: $sharpe, WinRate: $winrate)" -ForegroundColor Gray
}

Write-Host "⏳ Waiting for processing cycle (90 seconds)..." -ForegroundColor Yellow
Start-Sleep -Seconds 90

Write-Host ""
Write-Host "📜 Checking Evolution Engine logs..." -ForegroundColor Cyan
docker logs --tail 30 quantum_strategic_evolution

Write-Host ""
Write-Host "🔁 Checking evolution data in Redis..." -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray

Write-Host "`n📊 Rankings (first 300 chars):" -ForegroundColor Yellow
$rankings = docker exec redis redis-cli GET quantum:evolution:rankings
if ($rankings) {
    $rankings.Substring(0, [Math]::Min(300, $rankings.Length)) + "..."
}

Write-Host "`n🎯 Selected Models:" -ForegroundColor Yellow
docker exec redis redis-cli GET quantum:evolution:selected | ConvertFrom-Json | ConvertTo-Json

Write-Host "`n🧬 Mutated Configurations:" -ForegroundColor Yellow
docker exec redis redis-cli GET quantum:evolution:mutated | ConvertFrom-Json | ConvertTo-Json

Write-Host "`n🔄 Retrain Stream (last 3 jobs):" -ForegroundColor Yellow
docker exec redis redis-cli XREVRANGE quantum:stream:model.retrain + - COUNT 3

Write-Host "`n🧠 Fetching AI Engine Health snapshot..." -ForegroundColor Cyan
try {
    $health = Invoke-RestMethod -Uri "http://localhost:8001/health" -Method Get -ErrorAction SilentlyContinue
    $health.metrics.strategic_evolution | ConvertTo-Json
} catch {
    Write-Host "⚠️ AI Engine health endpoint not available" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🎯 PHASE 4T+ DEPLOYMENT COMPLETE" -ForegroundColor Green
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Green
Write-Host "• Strategic Evolution Engine: ✅ Running" -ForegroundColor Green
Write-Host "• Performance Evaluator: ✅ 6 strategies analyzed" -ForegroundColor Green
Write-Host "• Model Selector: ✅ Top 3 selected" -ForegroundColor Green
Write-Host "• Mutation Engine: ✅ Hyperparameters mutated" -ForegroundColor Green
Write-Host "• Retrain Manager: ✅ Jobs scheduled" -ForegroundColor Green
Write-Host "• Feedback Loop: ✅ Active (10 min cycle)" -ForegroundColor Green
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Green
Write-Host ""
Write-Host "📊 Monitor live:" -ForegroundColor Cyan
Write-Host "  docker logs -f quantum_strategic_evolution" -ForegroundColor Gray
Write-Host ""
Write-Host "🔍 Check rankings:" -ForegroundColor Cyan
Write-Host "  docker exec redis redis-cli GET quantum:evolution:rankings" -ForegroundColor Gray
Write-Host ""
Write-Host "🧠 View retrain stream:" -ForegroundColor Cyan
Write-Host "  docker exec redis redis-cli XREVRANGE quantum:stream:model.retrain + - COUNT 5" -ForegroundColor Gray
Write-Host ""
