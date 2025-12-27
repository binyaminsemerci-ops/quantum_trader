# ============================================================
# QUANTUM TRADER - Complete System Startup Script
# ============================================================
# This script starts all system components:
# 1. Docker containers (optional)
# 2. Backend API server
# 3. Frontend dashboard
# 4. Continuous Learning Manager (CML)
# 5. AI Model Training (optional)
# ============================================================

param(
    [switch]$SkipDocker,
    [switch]$SkipTraining,
    [switch]$TrainingOnly
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSCommandPath

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "🚀 QUANTUM TRADER - SYSTEM STARTUP" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# Activate virtual environment
Write-Host "[1/6] 🔧 Activating Python environment..." -ForegroundColor Yellow
$venvActivate = Join-Path $root ".venv/Scripts/Activate.ps1"
if (Test-Path $venvActivate) {
    . $venvActivate
    Write-Host "     ✅ Python environment activated" -ForegroundColor Green
} else {
    Write-Host "     ⚠️  No virtual environment found" -ForegroundColor Yellow
}

# Start Docker (optional)
if (-not $SkipDocker -and -not $TrainingOnly) {
    Write-Host ""
    Write-Host "[2/6] 🐳 Starting Docker containers..." -ForegroundColor Yellow
    Push-Location $root
    try {
        docker-compose --profile dev up -d 2>&1 | Out-Null
        Write-Host "     ✅ Docker containers started" -ForegroundColor Green
    } catch {
        Write-Host "     ⚠️  Docker not available or already running: $($_.Exception.Message)" -ForegroundColor Yellow
    }
    Pop-Location
} else {
    Write-Host ""
    Write-Host "[2/6] ⏭️  Skipping Docker (--SkipDocker or --TrainingOnly)" -ForegroundColor Gray
}

# Start Backend
if (-not $TrainingOnly) {
    Write-Host ""
    Write-Host "[3/6] 🖥️  Starting Backend server..." -ForegroundColor Yellow
    
    # Check if backend is already running
    try {
        $response = Invoke-WebRequest -UseBasicParsing "http://localhost:8000/health" -TimeoutSec 2 -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            Write-Host "     ✅ Backend already running on port 8000" -ForegroundColor Green
        }
    } catch {
        # Start backend in new window
        Write-Host "     ⏳ Launching backend server..."
        Start-Process -WorkingDirectory $root -FilePath "pwsh" -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "$root/scripts/start-backend.ps1", "-NewWindow" -WindowStyle Minimized
        Start-Sleep 5
        
        # Verify backend started
        try {
            $response = Invoke-WebRequest -UseBasicParsing "http://localhost:8000/health" -TimeoutSec 5
            Write-Host "     ✅ Backend started successfully" -ForegroundColor Green
        } catch {
            Write-Host "     ⚠️  Backend health check failed" -ForegroundColor Yellow
        }
    }
} else {
    Write-Host ""
    Write-Host "[3/6] ⏭️  Skipping Backend (--TrainingOnly)" -ForegroundColor Gray
}

# Start Frontend
if (-not $TrainingOnly) {
    Write-Host ""
    Write-Host "[4/6] 🌐 Starting Frontend dashboard..." -ForegroundColor Yellow
    
    # Check if frontend is already running
    try {
        $response = Invoke-WebRequest -UseBasicParsing "http://localhost:3000" -TimeoutSec 2 -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            Write-Host "     ✅ Frontend already running on port 3000" -ForegroundColor Green
        }
    } catch {
        # Start frontend in background
        Write-Host "     ⏳ Launching frontend server..."
        Push-Location "$root/frontend"
        Start-Process -FilePath "cmd" -ArgumentList "/c", "npm run dev" -WindowStyle Minimized
        Pop-Location
        Start-Sleep 3
        Write-Host "     ✅ Frontend started" -ForegroundColor Green
    }
} else {
    Write-Host ""
    Write-Host "[4/6] ⏭️  Skipping Frontend (--TrainingOnly)" -ForegroundColor Gray
}

# Activate Continuous Learning Manager
Write-Host ""
Write-Host "[5/6] 🧠 Activating Continuous Learning Manager..." -ForegroundColor Yellow
Push-Location $root
try {
    python activate_retraining_system.py | Out-Null
    Write-Host "     ✅ CML activated successfully" -ForegroundColor Green
} catch {
    Write-Host "     ⚠️  CML activation warning: $($_.Exception.Message)" -ForegroundColor Yellow
}
Pop-Location

# Train AI Models (optional)
if (-not $SkipTraining) {
    Write-Host ""
    Write-Host "[6/6] 🤖 Starting AI model training..." -ForegroundColor Yellow
    Write-Host "     ⏳ This may take 30-40 minutes..." -ForegroundColor Gray
    Write-Host ""
    
    Push-Location $root
    # Run training in current window if TrainingOnly, otherwise in background
    if ($TrainingOnly) {
        python scripts/train_all_models.py
    } else {
        Start-Process -FilePath "pwsh" -ArgumentList "-NoProfile", "-Command", "cd '$root'; & '.venv/Scripts/Activate.ps1'; python scripts/train_all_models.py; Read-Host 'Press Enter to close'" -WindowStyle Normal
        Write-Host "     ✅ Training started in separate window" -ForegroundColor Green
    }
    Pop-Location
} else {
    Write-Host ""
    Write-Host "[6/6] ⏭️  Skipping AI training (--SkipTraining)" -ForegroundColor Gray
}

# Summary
Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "✅ STARTUP COMPLETE!" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "📊 System Status:" -ForegroundColor White
if (-not $TrainingOnly) {
    Write-Host "   • Backend:  http://localhost:8000" -ForegroundColor White
    Write-Host "   • Frontend: http://localhost:3000" -ForegroundColor White
    Write-Host "   • Health:   http://localhost:8000/health" -ForegroundColor White
}
Write-Host "   • CML:      Activated ✅" -ForegroundColor White
if (-not $SkipTraining) {
    Write-Host "   • Training: In progress 🔄" -ForegroundColor White
}
Write-Host ""
Write-Host "🎯 Quick Commands:" -ForegroundColor White
Write-Host "   • Check backend:  Invoke-WebRequest http://localhost:8000/health" -ForegroundColor Gray
Write-Host "   • Check frontend: Invoke-WebRequest http://localhost:3000" -ForegroundColor Gray
Write-Host "   • View logs:      Get-Content backend_startup.log -Tail 50 -Wait" -ForegroundColor Gray
Write-Host ""
Write-Host "⚙️  Script Options:" -ForegroundColor White
Write-Host "   • .\startup_all.ps1               # Full startup" -ForegroundColor Gray
Write-Host "   • .\startup_all.ps1 -SkipDocker   # Skip Docker" -ForegroundColor Gray
Write-Host "   • .\startup_all.ps1 -SkipTraining # Skip AI training" -ForegroundColor Gray
Write-Host "   • .\startup_all.ps1 -TrainingOnly # Only train models" -ForegroundColor Gray
Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
