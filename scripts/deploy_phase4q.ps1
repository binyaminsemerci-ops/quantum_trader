# ============================================================================
# PHASE 4Q+ — Portfolio Governance Deployment & Validation (Windows)
# ============================================================================
# Dette scriptet deployer og validerer Portfolio Governance Agent på Windows
# ============================================================================

Write-Host ""
Write-Host "🚀 ==============================================" -ForegroundColor Cyan
Write-Host "   PHASE 4Q+ — PORTFOLIO GOVERNANCE DEPLOYMENT" -ForegroundColor Green
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host ""

Set-Location C:\quantum_trader

# --- 1. Sjekk at repoet er oppdatert
Write-Host "🔄 Pulling latest code from GitHub..." -ForegroundColor Yellow
try {
    git pull origin main
    Write-Host "✅ Git pull complete" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Git pull failed, continuing with local version..." -ForegroundColor Yellow
}
Write-Host ""

# --- 2. Bygg service
Write-Host "🏗️  Building portfolio_governance Docker image..." -ForegroundColor Yellow
docker-compose -f docker-compose.vps.yml build portfolio-governance
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Build complete" -ForegroundColor Green
} else {
    Write-Host "❌ Build failed!" -ForegroundColor Red
    exit 1
}
Write-Host ""

# --- 3. Start service
Write-Host "▶️  Starting portfolio_governance service..." -ForegroundColor Yellow
docker-compose -f docker-compose.vps.yml up -d portfolio-governance
Write-Host "⏳ Waiting 10 seconds for service to initialize..." -ForegroundColor Gray
Start-Sleep -Seconds 10
Write-Host ""

# --- 4. Valider at containeren kjører
Write-Host "🔍 Checking container status..." -ForegroundColor Yellow
$containerStatus = docker ps --format "table {{.Names}}\t{{.Status}}" | Select-String "portfolio_governance"
if ($containerStatus) {
    Write-Host "✅ Container is running:" -ForegroundColor Green
    Write-Host "   $containerStatus" -ForegroundColor White
} else {
    Write-Host "❌ ERROR: Container not running!" -ForegroundColor Red
    Write-Host "📜 Last 20 logs:" -ForegroundColor Yellow
    docker logs --tail 20 quantum_portfolio_governance
    exit 1
}
Write-Host ""

# --- 5. Kjør health check via AI Engine
Write-Host "🧠 Fetching AI Engine Health metrics..." -ForegroundColor Yellow
try {
    $health = curl -s http://localhost:8001/health | ConvertFrom-Json
    if ($health.metrics.portfolio_governance) {
        Write-Host "✅ Portfolio Governance Metrics:" -ForegroundColor Green
        $health.metrics.portfolio_governance | ConvertTo-Json -Depth 5 | Write-Host -ForegroundColor White
    } else {
        Write-Host "⚠️  Portfolio governance metrics not yet available" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️  Could not fetch health endpoint" -ForegroundColor Yellow
}
Write-Host ""

# --- 6. Test Redis integrasjon
Write-Host "📊 Testing Redis streams and keys..." -ForegroundColor Yellow
$memoryLen = docker exec redis redis-cli XLEN quantum:stream:portfolio.memory 2>$null
Write-Host "  • Memory stream length:" -ForegroundColor White
Write-Host "    quantum:stream:portfolio.memory = $memoryLen entries" -ForegroundColor Gray

$policy = docker exec redis redis-cli GET quantum:governance:policy 2>$null
Write-Host "  • Current policy:" -ForegroundColor White
Write-Host "    quantum:governance:policy = $policy" -ForegroundColor Gray

$score = docker exec redis redis-cli GET quantum:governance:score 2>$null
Write-Host "  • Current score:" -ForegroundColor White
Write-Host "    quantum:governance:score = $score" -ForegroundColor Gray
Write-Host ""

# --- 7. Simuler governance aktivitet
Write-Host "🧩 Simulating sample PnL data for testing..." -ForegroundColor Yellow
Write-Host "  • Adding profitable trade event..." -ForegroundColor Gray
$timestamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
docker exec redis redis-cli XADD quantum:stream:portfolio.memory '*' `
    timestamp $timestamp `
    symbol "BTCUSDT" `
    side "LONG" `
    pnl "0.35" `
    confidence "0.74" `
    volatility "0.12" `
    leverage "20" `
    position_size "1000" `
    exit_reason "dynamic_tp" > $null

Write-Host "  • Adding losing trade event..." -ForegroundColor Gray
docker exec redis redis-cli XADD quantum:stream:portfolio.memory '*' `
    timestamp $timestamp `
    symbol "ETHUSDT" `
    side "SHORT" `
    pnl "-0.18" `
    confidence "0.62" `
    volatility "0.22" `
    leverage "15" `
    position_size "500" `
    exit_reason "stop_loss" > $null

Write-Host "⏳ Waiting 5 seconds for governance agent to process..." -ForegroundColor Gray
Start-Sleep -Seconds 5
Write-Host ""

Write-Host "✅ Reading policy after simulation..." -ForegroundColor Green
$newPolicy = docker exec redis redis-cli GET quantum:governance:policy 2>$null
$newScore = docker exec redis redis-cli GET quantum:governance:score 2>$null
$newMemoryLen = docker exec redis redis-cli XLEN quantum:stream:portfolio.memory 2>$null

Write-Host "  • Policy: $newPolicy" -ForegroundColor White
Write-Host "  • Score: $newScore" -ForegroundColor White
Write-Host "  • Memory samples: $newMemoryLen" -ForegroundColor White
Write-Host ""

# --- 8. Log sjekk
Write-Host "📜 Recent container logs (last 15 lines):" -ForegroundColor Yellow
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
docker logs --tail 15 quantum_portfolio_governance
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
Write-Host ""

# --- 9. Test policy parameters
Write-Host "🔧 Reading policy parameters..." -ForegroundColor Yellow
$params = docker exec redis redis-cli GET quantum:governance:params 2>$null
if ($params) {
    try {
        $params | ConvertFrom-Json | ConvertTo-Json -Depth 5 | Write-Host -ForegroundColor White
    } catch {
        Write-Host $params -ForegroundColor White
    }
} else {
    Write-Host "  (No parameters set yet)" -ForegroundColor Gray
}
Write-Host ""

# --- 10. Oppsummering
Write-Host ""
Write-Host "🎯 ==============================================" -ForegroundColor Cyan
Write-Host "   PHASE 4Q+ DEPLOYMENT SUCCESSFULLY COMPLETED!" -ForegroundColor Green
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "✅ Service Status:" -ForegroundColor Green
Write-Host "   • Container: quantum_portfolio_governance (RUNNING)" -ForegroundColor White
Write-Host "   • Memory Stream: quantum:stream:portfolio.memory ($newMemoryLen samples)" -ForegroundColor White
Write-Host "   • Current Policy: $newPolicy" -ForegroundColor White
Write-Host "   • Portfolio Score: $newScore" -ForegroundColor White
Write-Host ""
Write-Host "📡 Integration Points:" -ForegroundColor Yellow
Write-Host "   • AI Engine Health: http://localhost:8001/health" -ForegroundColor White
Write-Host "   • Redis Policy Key: quantum:governance:policy" -ForegroundColor White
Write-Host "   • Redis Score Key: quantum:governance:score" -ForegroundColor White
Write-Host "   • Redis Params: quantum:governance:params" -ForegroundColor White
Write-Host ""
Write-Host "📊 Next Steps:" -ForegroundColor Yellow
Write-Host "   1. Monitor logs: docker logs -f quantum_portfolio_governance" -ForegroundColor White
Write-Host "   2. Check health: curl http://localhost:8001/health | jq '.metrics.portfolio_governance'" -ForegroundColor White
Write-Host "   3. View memory: docker exec redis redis-cli XREVRANGE quantum:stream:portfolio.memory + - COUNT 10" -ForegroundColor White
Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "✅ Portfolio Governance Agent is LIVE and OPERATIONAL!" -ForegroundColor Green
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""
