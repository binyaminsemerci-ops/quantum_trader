# Quantum Trader - PowerShell Startup Script
# ===========================================
# Starts all components with better error handling

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  QUANTUM TRADER - SYSTEM STARTUP" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

# Function to check if command exists
function Test-CommandExists {
    param($command)
    $null -ne (Get-Command $command -ErrorAction SilentlyContinue)
}

# [1/6] Check Docker
Write-Host "[1/6] Checking Docker..." -ForegroundColor Yellow
if (-not (Test-CommandExists docker)) {
    Write-Host "ERROR: Docker is not installed!" -ForegroundColor Red
    Write-Host "Please install Docker Desktop from: https://www.docker.com/products/docker-desktop" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

try {
    docker info | Out-Null
    Write-Host "OK: Docker is running" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Docker is not running!" -ForegroundColor Red
    Write-Host "Please start Docker Desktop first." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# [2/6] Start Docker containers
Write-Host ""
Write-Host "[2/6] Starting Docker containers..." -ForegroundColor Yellow
try {
    # Use docker-compose with profile
    $result = docker-compose --profile dev up -d 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Docker compose failed: $result"
    }
    Write-Host "OK: Containers started" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Failed to start containers" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
    Write-Host "Try running: docker-compose down -v && docker-compose --profile dev up -d" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# [3/6] Wait for backend
Write-Host ""
Write-Host "[3/6] Waiting for backend to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

$maxAttempts = 30
$attempt = 0
$backendReady = $false

while ($attempt -lt $maxAttempts) {
    $attempt++
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop
        if ($response.StatusCode -eq 200) {
            $backendReady = $true
            break
        }
    } catch {
        Write-Host "Waiting for backend... ($attempt/$maxAttempts)" -ForegroundColor Gray
        Start-Sleep -Seconds 2
    }
}

if ($backendReady) {
    Write-Host "OK: Backend is healthy" -ForegroundColor Green
} else {
    Write-Host "TIMEOUT: Backend did not start in time" -ForegroundColor Red
    Write-Host "Check logs with: docker logs quantum_backend" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# [4/6] Wait for frontend
Write-Host ""
Write-Host "[4/6] Waiting for frontend to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

$maxFrontendAttempts = 20
$frontendAttempt = 0
$frontendReady = $false

while ($frontendAttempt -lt $maxFrontendAttempts) {
    $frontendAttempt++
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:3000" -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop
        if ($response.StatusCode -eq 200) {
            $frontendReady = $true
            break
        }
    } catch {
        Write-Host "Waiting for frontend... ($frontendAttempt/$maxFrontendAttempts)" -ForegroundColor Gray
        Start-Sleep -Seconds 2
    }
}

if ($frontendReady) {
    Write-Host "OK: Frontend is ready" -ForegroundColor Green
} else {
    Write-Host "WARNING: Frontend not responding (check docker logs quantum_frontend)" -ForegroundColor Yellow
}

# [5/6] Verify Trading Profile
Write-Host ""
Write-Host "[5/6] Verifying Trading Profile..." -ForegroundColor Yellow
try {
    $config = Invoke-RestMethod -Uri "http://localhost:8000/trading-profile/config" -TimeoutSec 5
    if ($config.enabled) {
        Write-Host "OK: Trading Profile is ENABLED" -ForegroundColor Green
        Write-Host "   - Leverage: $($config.risk.default_leverage)x" -ForegroundColor Gray
        Write-Host "   - Max Positions: $($config.risk.max_positions)" -ForegroundColor Gray
        Write-Host "   - TP/SL: $($config.tpsl.atr_mult_tp1)R / $($config.tpsl.atr_mult_tp2)R" -ForegroundColor Gray
    } else {
        Write-Host "WARNING: Trading Profile is DISABLED" -ForegroundColor Yellow
    }
} catch {
    Write-Host "WARNING: Could not verify Trading Profile" -ForegroundColor Yellow
}

# [6/6] Open monitoring
Write-Host ""
Write-Host "[6/6] Opening monitoring dashboard..." -ForegroundColor Yellow
Start-Process python -ArgumentList "quick_status.py" -NoNewWindow -Wait

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  QUANTUM TRADER IS NOW RUNNING!" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Backend:  http://localhost:8000" -ForegroundColor White
Write-Host "Frontend: http://localhost:3000" -ForegroundColor White
Write-Host ""
Write-Host "Monitoring commands:" -ForegroundColor Yellow
Write-Host "  - Quick Status:   python quick_status.py" -ForegroundColor Gray
Write-Host "  - Live Monitor:   python monitor_production.py" -ForegroundColor Gray
Write-Host "  - Backend Logs:   docker logs -f quantum_backend" -ForegroundColor Gray
Write-Host ""
Write-Host "Press Enter to close this window..." -ForegroundColor Cyan
Read-Host
