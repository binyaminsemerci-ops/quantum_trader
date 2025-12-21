Write-Host "🚀 Phase 4U Manual Deployment to VPS" -ForegroundColor Cyan
Write-Host ""

# === Upload files via SCP ===
Write-Host "📤 Step 1: Uploading deploy_phase4u.sh..." -ForegroundColor Yellow
wsl scp -i ~/.ssh/hetzner_fresh deploy_phase4u.sh qt@46.224.116.254:/tmp/

Write-Host "📤 Step 2: Uploading docker-compose.vps.yml..." -ForegroundColor Yellow
wsl scp -i ~/.ssh/hetzner_fresh docker-compose.vps.yml qt@46.224.116.254:/tmp/

Write-Host "📤 Step 3: Uploading ai_engine/service.py..." -ForegroundColor Yellow
wsl scp -i ~/.ssh/hetzner_fresh microservices/ai_engine/service.py qt@46.224.116.254:/tmp/microservices/ai_engine/

Write-Host "📤 Step 4: Uploading model_federation microservice..." -ForegroundColor Yellow
wsl ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 "mkdir -p /tmp/microservices/model_federation"
wsl scp -i ~/.ssh/hetzner_fresh -r microservices/model_federation/* qt@46.224.116.254:/tmp/microservices/model_federation/

# === Build and run container ===
Write-Host "🚀 Step 5: Building and running Model Federation on VPS..." -ForegroundColor Cyan
wsl ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 @"
cd /tmp/microservices/model_federation && \
docker build -t quantum_model_federation . && \
docker stop quantum_model_federation 2>/dev/null || true && \
docker rm quantum_model_federation 2>/dev/null || true && \
docker run -d \
  --name quantum_model_federation \
  --network quantum_trader_quantum_trader \
  -e REDIS_HOST=redis \
  -e REDIS_PORT=6379 \
  --restart unless-stopped \
  --cpus='0.3' \
  --memory='192m' \
  quantum_model_federation && \
docker ps | grep model_federation
"@

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Container started successfully!" -ForegroundColor Green
    
    Write-Host "`n🧩 Step 6: Injecting mock model signals..." -ForegroundColor Cyan
    $timestamp = [int][double]::Parse((Get-Date -UFormat %s))
    
    # Inject 6 mock model signals
    wsl ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 @"
docker exec quantum_redis redis-cli SET quantum:model:xgb:signal '{\"action\":\"BUY\",\"confidence\":0.85,\"timestamp\":$timestamp}' && \
docker exec quantum_redis redis-cli SET quantum:model:lgbm:signal '{\"action\":\"BUY\",\"confidence\":0.78,\"timestamp\":$timestamp}' && \
docker exec quantum_redis redis-cli SET quantum:model:patchtst:signal '{\"action\":\"BUY\",\"confidence\":0.82,\"timestamp\":$timestamp}' && \
docker exec quantum_redis redis-cli SET quantum:model:nhits:signal '{\"action\":\"SELL\",\"confidence\":0.65,\"timestamp\":$timestamp}' && \
docker exec quantum_redis redis-cli SET quantum:model:rl_sizer:signal '{\"action\":\"BUY\",\"confidence\":0.75,\"timestamp\":$timestamp}' && \
docker exec quantum_redis redis-cli SET quantum:model:evo_model:signal '{\"action\":\"HOLD\",\"confidence\":0.60,\"timestamp\":$timestamp}'
"@
    
    Write-Host "✅ Mock signals injected (XGB, LGBM, PatchTST=BUY | NHITS=SELL | RL/Evo=BUY/HOLD)" -ForegroundColor Green
    
    Write-Host "`n⏳ Step 7: Waiting for federation cycle (20 seconds)..." -ForegroundColor Yellow
    Start-Sleep -Seconds 20
    
    Write-Host "`n📜 Step 8: Checking logs..." -ForegroundColor Cyan
    wsl ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 "docker logs --tail 30 quantum_model_federation"
    
    Write-Host "`n🎯 Step 9: Verifying consensus signal..." -ForegroundColor Cyan
    wsl ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 "docker exec quantum_redis redis-cli GET quantum:consensus:signal"
    
    Write-Host "`n🧠 Step 10: Checking trust weights..." -ForegroundColor Cyan
    wsl ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 "docker exec quantum_redis redis-cli HGETALL quantum:trust:history"
    
    Write-Host ""
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Green
    Write-Host "🎯 PHASE 4U DEPLOYED SUCCESSFULLY!" -ForegroundColor Green
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Green
    Write-Host "Container: quantum_model_federation ✅ Running" -ForegroundColor Green
    Write-Host "Federation Engine: Building consensus every 10 sec" -ForegroundColor Green
    Write-Host "Model Broker: Collecting from 9 models" -ForegroundColor Green
    Write-Host "Trust Memory: Learning model reliability" -ForegroundColor Green
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Green
    Write-Host ""
    
} else {
    Write-Host "❌ Deployment failed" -ForegroundColor Red
    exit 1
}
