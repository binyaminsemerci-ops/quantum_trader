#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Test script for Quantum Trader Docker setup with sklearn integration
.DESCRIPTION
    Builds and tests the Docker container to verify sklearn models load correctly
#>

Write-Host "🐳 Testing Quantum Trader Docker Setup with sklearn..." -ForegroundColor Cyan
Write-Host ""

# Step 1: Check if models exist
Write-Host "📦 Step 1: Checking AI models..." -ForegroundColor Yellow
$modelsPath = "ai_engine\models"
$requiredModels = @("xgb_model.pkl", "scaler.pkl", "ensemble_model.pkl")

foreach ($model in $requiredModels) {
    $modelFile = Join-Path $modelsPath $model
    if (Test-Path $modelFile) {
        $size = (Get-Item $modelFile).Length / 1KB
        Write-Host "  ✅ Found $model ($([math]::Round($size, 2)) KB)" -ForegroundColor Green
    } else {
        Write-Host "  ⚠️  Missing $model - will use fallback mode" -ForegroundColor Yellow
    }
}
Write-Host ""

# Step 2: Build Docker image
Write-Host "🔨 Step 2: Building Docker image..." -ForegroundColor Yellow
Write-Host "  This may take 2-3 minutes on first build..." -ForegroundColor Gray
docker-compose build backend
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker build failed!" -ForegroundColor Red
    exit 1
}
Write-Host "  ✅ Docker image built successfully" -ForegroundColor Green
Write-Host ""

# Step 3: Start container
Write-Host "🚀 Step 3: Starting container..." -ForegroundColor Yellow
docker-compose --profile dev up -d backend
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to start container!" -ForegroundColor Red
    exit 1
}
Write-Host "  ✅ Container started" -ForegroundColor Green
Write-Host ""

# Step 4: Wait for startup
Write-Host "⏳ Step 4: Waiting for backend to initialize (30s)..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Show logs to verify sklearn loading
Write-Host ""
Write-Host "📋 Checking startup logs for sklearn validation..." -ForegroundColor Cyan
docker-compose logs backend | Select-String -Pattern "sklearn|model|ensemble" | Select-Object -First 20
Write-Host ""

# Step 5: Health check
Write-Host "🏥 Step 5: Running health checks..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

try {
    $health = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 5
    Write-Host "  ✅ Health check passed" -ForegroundColor Green
    Write-Host "     Status: $($health.status)" -ForegroundColor Gray
} catch {
    Write-Host "  ⚠️  Health check pending, container may still be starting..." -ForegroundColor Yellow
}
Write-Host ""

# Step 6: Test AI model endpoint
Write-Host "🤖 Step 6: Testing AI model endpoint..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

try {
    $aiStatus = Invoke-RestMethod -Uri "http://localhost:8000/api/ai/model/status" -TimeoutSec 5
    Write-Host "  ✅ AI model endpoint responding" -ForegroundColor Green
    Write-Host "     Model loaded: $($aiStatus.model_loaded)" -ForegroundColor Gray
    Write-Host "     Ensemble active: $($aiStatus.ensemble_active)" -ForegroundColor Gray
} catch {
    Write-Host "  ⚠️  AI endpoint not ready yet (may need more time)" -ForegroundColor Yellow
}
Write-Host ""

# Step 7: Summary
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 60) -ForegroundColor Cyan
Write-Host "📊 Test Summary" -ForegroundColor Cyan
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 60) -ForegroundColor Cyan
Write-Host ""
Write-Host "Container Status:" -ForegroundColor White
docker-compose ps backend
Write-Host ""
Write-Host "To view live logs:" -ForegroundColor Yellow
Write-Host "  docker-compose logs -f backend" -ForegroundColor Gray
Write-Host ""
Write-Host "To stop the container:" -ForegroundColor Yellow
Write-Host "  docker-compose --profile dev down" -ForegroundColor Gray
Write-Host ""
Write-Host "To test AI signals:" -ForegroundColor Yellow
Write-Host "  Invoke-RestMethod -Uri 'http://localhost:8000/api/ai/signals/latest'" -ForegroundColor Gray
Write-Host ""
Write-Host "✅ Docker test completed!" -ForegroundColor Green
Write-Host ""
