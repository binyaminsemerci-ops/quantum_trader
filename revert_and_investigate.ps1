Write-Host "`n=== REVERTING MES ENDRINGER ===" -ForegroundColor Red
cd c:\quantum_trader

# Revert executor_service.py
git checkout backend/microservices/auto_executor/executor_service.py
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Reverted executor_service.py" -ForegroundColor Green
}

Write-Host "`n=== NÅ UNDERSØKER JEG SYSTEMET ORDENTLIG ===" -ForegroundColor Cyan
Write-Host ""

# 1. Sjekk hvilke containers som kjører på VPS
Write-Host "1️⃣ Sjekker aktive containers på VPS..." -ForegroundColor Yellow
ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 "docker ps --format 'table {{.Names}}\t{{.Status}}'"

Write-Host "`n2️⃣ Sjekker om TradeIntentSubscriber kjører..." -ForegroundColor Yellow
ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 "docker logs quantum_ai_engine --tail 100 | grep -i 'TradeIntent\|trade.intent\|Subscribed' || echo 'Ingen TradeIntentSubscriber logs'"

Write-Host "`n3️⃣ Sjekker EventBus stream for trade.intent..." -ForegroundColor Yellow  
ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 "docker exec quantum_redis redis-cli XLEN quantum:stream:trade.intent 2>/dev/null || echo 'Stream finnes ikke'"

Write-Host "`n4️⃣ Sjekker om Trading Bot publiserer signaler..." -ForegroundColor Yellow
ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 "docker logs quantum_trading_bot --tail 50 | grep -i 'Published trade.intent' | tail -5 || echo 'Ingen publiserte signaler'"

Write-Host "`n5️⃣ Sjekker Auto Executor status..." -ForegroundColor Yellow
ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 "docker logs quantum_auto_executor --tail 30"

Write-Host "`n=== ANALYSE FERDIG ===" -ForegroundColor Green
