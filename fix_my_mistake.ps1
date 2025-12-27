#!/usr/bin/env pwsh
Write-Host "🔧 REVERTING MINE FEIL ENDRINGER..." -ForegroundColor Red
Write-Host ""

# SSH til VPS og revert
ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 @"
cd /home/qt/quantum_trader
echo '1️⃣ Reverting executor_service.py fra git...'
git checkout backend/microservices/auto_executor/executor_service.py
echo '✅ Reverted'
echo ''
echo '2️⃣ Kopierer tilbake til container...'
docker cp backend/microservices/auto_executor/executor_service.py quantum_auto_executor:/app/
echo '✅ Copied'
echo ''
echo '3️⃣ Restarting quantum_auto_executor...'
docker restart quantum_auto_executor
echo '✅ Restarted'
echo ''
echo '⏳ Venter 5 sekunder...'
sleep 5
echo ''
echo '📋 Sjekker logs...'
docker logs quantum_auto_executor --tail 20
"@

Write-Host ""
Write-Host "✅ FEIL RETTET OPP!" -ForegroundColor Green
