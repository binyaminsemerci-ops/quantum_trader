# Phase 2 Deployment Script for PowerShell
# ===========================================

$VPS_IP = "46.224.116.254"
$VPS_USER = "benyamin"

Write-Host "════════════════════════════════════════════════" -ForegroundColor Blue
Write-Host "⚡ PHASE 2: CIRCUIT BREAKER + REDIS FIX" -ForegroundColor Blue
Write-Host "════════════════════════════════════════════════" -ForegroundColor Blue
Write-Host ""

# STEG 1: GIT PULL
Write-Host "1️⃣ Git pull..." -ForegroundColor Yellow
ssh "$VPS_USER@$VPS_IP" "cd quantum_trader && git pull origin main"
Write-Host "✅ Git pull done" -ForegroundColor Green

# STEG 2: REBUILD
Write-Host ""
Write-Host "2️⃣ Rebuilding containers..." -ForegroundColor Yellow
ssh "$VPS_USER@$VPS_IP" "cd quantum_trader && docker compose -f docker-compose.vps.yml build backend ai_engine"
Write-Host "✅ Containers rebuilt" -ForegroundColor Green

# STEG 3: RESTART
Write-Host ""
Write-Host "3️⃣ Restarting services..." -ForegroundColor Yellow
ssh "$VPS_USER@$VPS_IP" "cd quantum_trader && docker compose -f docker-compose.vps.yml restart backend cross_exchange eventbus_bridge"
Write-Host "✅ Services restarted" -ForegroundColor Green

# STEG 4: WAIT
Write-Host ""
Write-Host "4️⃣ Waiting for startup (15 seconds)..." -ForegroundColor Yellow
Start-Sleep -Seconds 15

# STEG 5: TEST
Write-Host ""
Write-Host "5️⃣ Testing Circuit Breaker API..." -ForegroundColor Yellow
ssh "$VPS_USER@$VPS_IP" "curl -s http://localhost:8000/api/circuit-breaker/status | jq '.'"

# STEG 6: LOGS
Write-Host ""
Write-Host "6️⃣ Checking logs..." -ForegroundColor Yellow
Write-Host "=== BACKEND ===" -ForegroundColor Cyan
ssh "$VPS_USER@$VPS_IP" "docker logs quantum_backend 2>&1 | tail -15 | grep -E '(Circuit|Redis)'"

Write-Host ""
Write-Host "=== CROSS EXCHANGE ===" -ForegroundColor Cyan
ssh "$VPS_USER@$VPS_IP" "docker logs quantum_cross_exchange 2>&1 | tail -15 | grep -E '(Redis|Connected)'"

Write-Host ""
Write-Host "════════════════════════════════════════════════" -ForegroundColor Green
Write-Host "✅ PHASE 2 DEPLOYMENT COMPLETE!" -ForegroundColor Green
Write-Host "════════════════════════════════════════════════" -ForegroundColor Green
Write-Host ""
Write-Host "Test Circuit Breaker API:" -ForegroundColor Blue
Write-Host "curl http://46.224.116.254:8000/api/circuit-breaker/status"
