# Test AI Engine Health Endpoint
Write-Host "`n==================================" -ForegroundColor Cyan
Write-Host "AI ENGINE HEALTH TEST" -ForegroundColor Cyan
Write-Host "==================================`n" -ForegroundColor Cyan

# Start service in WSL (background)
Write-Host "[1/3] Starting AI Engine service in WSL..." -ForegroundColor Yellow
wsl bash -c 'cd ~/quantum_trader && pkill -9 -f "uvicorn.*ai_engine" 2>/dev/null; source .venv/bin/activate && export PYTHONPATH=$HOME/quantum_trader REDIS_HOST=localhost REDIS_PORT=6379 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 OMP_NUM_THREADS=2 && nohup uvicorn microservices.ai_engine.main:app --host 0.0.0.0 --port 8001 > /tmp/ai_engine.log 2>&1 & echo $!'

# Wait for startup
Write-Host "`n[2/3] Waiting for service to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Test health endpoint
Write-Host "`n[3/3] Testing health endpoint..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "http://localhost:8001/health" -Method Get -TimeoutSec 5
    Write-Host "`n✅ Health endpoint response:" -ForegroundColor Green
    $response | ConvertTo-Json -Depth 10
    
    if ($response.error) {
        Write-Host "`n⚠️  Error found: $($response.error)" -ForegroundColor Yellow
        Write-Host "`nChecking service logs..." -ForegroundColor Yellow
        wsl bash -c 'tail -50 /tmp/ai_engine.log | grep -E "Health check|ERROR|create"'
    } else {
        Write-Host "`n✅ Health check OK - No errors!" -ForegroundColor Green
    }
} catch {
    Write-Host "`n❌ Failed to reach health endpoint: $_" -ForegroundColor Red
    Write-Host "`nChecking if service is running..." -ForegroundColor Yellow
    wsl bash -c 'ps aux | grep uvicorn | grep -v grep'
    Write-Host "`nLast 20 lines of log:" -ForegroundColor Yellow
    wsl bash -c 'tail -20 /tmp/ai_engine.log'
}

Write-Host "`n==================================`n" -ForegroundColor Cyan
