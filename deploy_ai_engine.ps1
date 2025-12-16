# Deploy and test AI Engine (Windows PowerShell version)

Write-Host "`n╔═════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║   AI ENGINE - DEPLOY & TEST                 ║" -ForegroundColor Green  
Write-Host "╚═════════════════════════════════════════════╝`n" -ForegroundColor Green

# Step 1: Copy updated files
Write-Host "[1/5] Copying updated service.py to WSL..." -ForegroundColor Yellow
wsl bash -c 'cp /mnt/c/quantum_trader/microservices/ai_engine/service.py ~/quantum_trader/microservices/ai_engine/' 2>&1 | Out-Null
Write-Host "  ✅ Done" -ForegroundColor Green

# Step 2: Kill existing service
Write-Host "`n[2/5] Stopping existing service..." -ForegroundColor Yellow
wsl bash -c 'pkill -9 -f "uvicorn.*ai_engine" 2>/dev/null; sleep 1' 2>&1 | Out-Null
Write-Host "  ✅ Done" -ForegroundColor Green

# Step 3: Clean cache
Write-Host "`n[3/5] Cleaning Python cache..." -ForegroundColor Yellow
wsl bash -c 'find ~/quantum_trader/microservices/ai_engine -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null' 2>&1 | Out-Null
Write-Host "  ✅ Done" -ForegroundColor Green

# Step 4: Start service using startup script  
Write-Host "`n[4/5] Starting AI Engine service..." -ForegroundColor Yellow
$startCmd = @"
cd ~/quantum_trader
source .venv/bin/activate
export PYTHONPATH=\$HOME/quantum_trader
export REDIS_HOST=localhost
export REDIS_PORT=6379
nohup uvicorn microservices.ai_engine.main:app --host 0.0.0.0 --port 8001 > /tmp/ai_engine.log 2>&1 &
echo \$!
"@
$pid = wsl bash -c $startCmd
Write-Host "  ✅ Service started with PID: $pid" -ForegroundColor Green

# Step 5: Wait and test
Write-Host "`n[5/5] Waiting for service to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 6

Write-Host "`nTesting health endpoint..." -ForegroundColor Cyan
try {
    $health = Invoke-RestMethod -Uri "http://localhost:8001/health" -Method Get -TimeoutSec 5
    
    Write-Host "`n╔═════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "║           HEALTH ENDPOINT RESPONSE          ║" -ForegroundColor Cyan
    Write-Host "╚═════════════════════════════════════════════╝" -ForegroundColor Cyan
    $health | ConvertTo-Json | Write-Host
    
    if ($health.error -eq "create" -or $health.status -eq "DEGRADED") {
        Write-Host "`n⚠️  Issue detected - Checking logs for details..." -ForegroundColor Yellow
        Write-Host "`n--- Service Logs ---" -ForegroundColor Cyan
        wsl bash -c 'tail -40 /tmp/ai_engine.log | grep -A 5 -B 5 -E "(ServiceHealth|Health check|create|ERROR)"'
    } else {
        Write-Host "`n✅ SERVICE HEALTHY - No errors!" -ForegroundColor Green
    }
    
} catch {
    Write-Host "`n❌ Failed to connect: $_" -ForegroundColor Red
    Write-Host "`nService logs:" -ForegroundColor Yellow
    wsl bash -c 'tail -50 /tmp/ai_engine.log'
}

Write-Host "`n╚═════════════════════════════════════════════╝`n" -ForegroundColor Green
