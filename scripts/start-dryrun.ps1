# 🧪 Hybrid Agent Dry-Run Quick Start Script
# Sets up paper trading mode and starts the system

Write-Host "`n🧪 HYBRID AGENT DRY-RUN SETUP" -ForegroundColor Cyan
Write-Host "="*80 -ForegroundColor Cyan

# Step 1: Check if build is complete
Write-Host "`n📦 Checking Docker image..." -ForegroundColor Yellow
$image = docker image ls quantum_trader-backend --format "{{.Size}}"
Write-Host "   Current image size: $image" -ForegroundColor White

if ($image -like "*3.87GB*") {
    Write-Host "   ⚠️  Old image detected (without PyTorch)" -ForegroundColor Yellow
    Write-Host "   Build may still be in progress..." -ForegroundColor Yellow
    
    $continue = Read-Host "`nContinue anyway? (y/n)"
    if ($continue -ne 'y') {
        Write-Host "`nℹ️  Exiting. Wait for build to complete." -ForegroundColor Blue
        exit 0
    }
} else {
    Write-Host "   ✅ New image detected (with PyTorch)" -ForegroundColor Green
}

# Step 2: Update .env for paper trading
Write-Host "`n🔧 Configuring paper trading mode..." -ForegroundColor Yellow

$envPath = ".env"
if (Test-Path $envPath) {
    $envContent = Get-Content $envPath -Raw
    
    # Enable paper trading
    if ($envContent -match 'QT_PAPER_TRADING=false') {
        $envContent = $envContent -replace 'QT_PAPER_TRADING=false', 'QT_PAPER_TRADING=true'
        Write-Host "   ✅ Set QT_PAPER_TRADING=true" -ForegroundColor Green
    } elseif ($envContent -notmatch 'QT_PAPER_TRADING') {
        $envContent += "`nQT_PAPER_TRADING=true"
        Write-Host "   ✅ Added QT_PAPER_TRADING=true" -ForegroundColor Green
    } else {
        Write-Host "   ℹ️  QT_PAPER_TRADING already set to true" -ForegroundColor Blue
    }
    
    # Verify confidence threshold
    if ($envContent -match 'QT_CONFIDENCE_THRESHOLD=(\d+\.?\d*)') {
        $threshold = $Matches[1]
        Write-Host "   ℹ️  QT_CONFIDENCE_THRESHOLD=$threshold" -ForegroundColor Blue
    } else {
        Write-Host "   ⚠️  No confidence threshold set, using default" -ForegroundColor Yellow
    }
    
    Set-Content -Path $envPath -Value $envContent -NoNewline
    Write-Host "   ✅ .env updated" -ForegroundColor Green
} else {
    Write-Host "   ❌ .env file not found!" -ForegroundColor Red
    exit 1
}

# Step 3: Stop existing backend
Write-Host "`n🛑 Stopping existing backend..." -ForegroundColor Yellow
docker-compose stop backend 2>&1 | Out-Null
Start-Sleep -Seconds 2
Write-Host "   ✅ Backend stopped" -ForegroundColor Green

# Step 4: Start backend in dry-run mode
Write-Host "`n🚀 Starting backend in DRY-RUN mode..." -ForegroundColor Yellow
docker-compose up -d backend

if ($LASTEXITCODE -eq 0) {
    Write-Host "   ✅ Backend starting..." -ForegroundColor Green
    
    # Wait for startup
    Write-Host "`n⏳ Waiting for backend to initialize (15 seconds)..." -ForegroundColor Yellow
    Start-Sleep -Seconds 15
    
    # Step 5: Check health
    Write-Host "`n🏥 Checking backend health..." -ForegroundColor Yellow
    try {
        $health = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 5
        Write-Host "   ✅ Backend is healthy!" -ForegroundColor Green
    } catch {
        Write-Host "   ⚠️  Backend not responding yet (may still be starting)" -ForegroundColor Yellow
    }
    
    # Step 6: Show initial logs
    Write-Host "`n📋 Initial logs (checking for Hybrid Agent)..." -ForegroundColor Yellow
    Write-Host "="*80 -ForegroundColor DarkGray
    docker logs quantum_backend --tail 30 2>&1 | Select-String -Pattern "Hybrid|TFT|XGBoost|confidence|ERROR" -CaseSensitive:$false
    Write-Host "="*80 -ForegroundColor DarkGray
    
    # Step 7: Run test suite
    Write-Host "`n🧪 Ready to run test suite!" -ForegroundColor Cyan
    Write-Host "`nTo run comprehensive tests:" -ForegroundColor White
    Write-Host "   python test_hybrid_dryrun.py" -ForegroundColor Green
    Write-Host "`nTo monitor live logs:" -ForegroundColor White
    Write-Host "   docker logs quantum_backend -f --tail 100" -ForegroundColor Green
    Write-Host "`nTo check for AI signals:" -ForegroundColor White
    Write-Host "   docker logs quantum_backend | Select-String 'AI Signal|confidence'" -ForegroundColor Green
    Write-Host "`nTo check positions:" -ForegroundColor White
    Write-Host "   python check_balance.py" -ForegroundColor Green
    Write-Host "`n" -ForegroundColor White
    
    # Offer to run test immediately
    $runTest = Read-Host "Run test suite now? (y/n)"
    if ($runTest -eq 'y') {
        Write-Host "`n🧪 Running test suite..." -ForegroundColor Cyan
        python test_hybrid_dryrun.py
    }
    
} else {
    Write-Host "   ❌ Failed to start backend!" -ForegroundColor Red
    Write-Host "`nCheck logs with: docker logs quantum_backend" -ForegroundColor Yellow
    exit 1
}
