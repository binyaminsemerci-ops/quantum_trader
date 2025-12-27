# Shadow Models Deployment Script
# This script automates Phase 1-2 of deployment

param(
    [switch]$TestMode,
    [switch]$SkipBackup
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   SHADOW MODELS DEPLOYMENT" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if running from workspace root
if (-not (Test-Path "ai_engine\ensemble_manager.py")) {
    Write-Host "Error: Must run from workspace root (c:\quantum_trader)" -ForegroundColor Red
    exit 1
}

# Phase 1: Pre-flight checks
Write-Host "[Phase 1] Pre-flight checks..." -ForegroundColor Yellow

# Check Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "  Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "  Python not found!" -ForegroundColor Red
    exit 1
}

# Check dependencies
Write-Host "`n  Checking dependencies..." -ForegroundColor Yellow
$required = @('numpy', 'scipy', 'pandas')
foreach ($pkg in $required) {
    $installed = python -c "import $pkg; print('OK')" 2>$null
    if ($installed -eq 'OK') {
        Write-Host "    $pkg OK" -ForegroundColor Green
    } else {
        Write-Host "    $pkg MISSING - installing..." -ForegroundColor Yellow
        pip install $pkg
    }
}

# Check if shadow_model_manager.py exists
if (-not (Test-Path "backend\services\ai\shadow_model_manager.py")) {
    Write-Host "`n  Error: shadow_model_manager.py not found!" -ForegroundColor Red
    Write-Host "  Please ensure Module 5 implementation is complete." -ForegroundColor Red
    exit 1
}
Write-Host "  shadow_model_manager.py found" -ForegroundColor Green

# Check if integration code exists
if (-not (Test-Path "backend\services\ai\shadow_model_integration.py")) {
    Write-Host "`n  Error: shadow_model_integration.py not found!" -ForegroundColor Red
    Write-Host "  Run deployment preparation first." -ForegroundColor Red
    exit 1
}
Write-Host "  shadow_model_integration.py found" -ForegroundColor Green

# Phase 2: Backup
if (-not $SkipBackup) {
    Write-Host "`n[Phase 2] Creating backups..." -ForegroundColor Yellow
    
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $backupDir = "backups\shadow_deployment_$timestamp"
    
    New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
    
    # Backup files
    Copy-Item "ai_engine\ensemble_manager.py" "$backupDir\ensemble_manager.py.bak"
    Copy-Item "backend\routes\ai.py" "$backupDir\ai.py.bak"
    
    Write-Host "  Backups created in $backupDir" -ForegroundColor Green
} else {
    Write-Host "`n[Phase 2] Skipping backup (--SkipBackup flag)" -ForegroundColor Yellow
}

# Phase 3: Environment configuration
Write-Host "`n[Phase 3] Configuring environment..." -ForegroundColor Yellow

$envPath = ".env"
$envConfig = @"

# Shadow Model Configuration (added by deploy_shadow_models.ps1)
ENABLE_SHADOW_MODELS=false
SHADOW_MIN_TRADES=500
SHADOW_MDD_TOLERANCE=1.20
SHADOW_ALPHA=0.05
SHADOW_N_BOOTSTRAP=10000
SHADOW_CHECK_INTERVAL=100
"@

if (Test-Path $envPath) {
    # Check if already configured
    $envContent = Get-Content $envPath -Raw
    if ($envContent -notmatch "ENABLE_SHADOW_MODELS") {
        Add-Content -Path $envPath -Value $envConfig
        Write-Host "  Added shadow config to .env" -ForegroundColor Green
    } else {
        Write-Host "  Shadow config already exists in .env" -ForegroundColor Yellow
    }
} else {
    $envConfig | Out-File -FilePath $envPath -Encoding UTF8
    Write-Host "  Created .env with shadow config" -ForegroundColor Green
}

# Phase 4: Integration instructions
Write-Host "`n[Phase 4] Integration required..." -ForegroundColor Yellow
Write-Host ""
Write-Host "  MANUAL STEPS REQUIRED:" -ForegroundColor Cyan
Write-Host "  =====================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  1. Open: ai_engine\ensemble_manager.py" -ForegroundColor White
Write-Host "     Copy sections from: backend\services\ai\shadow_model_integration.py" -ForegroundColor White
Write-Host "     - Imports (Part 1, lines 13-21)" -ForegroundColor Gray
Write-Host "     - __init__ code (Part 1, lines 24-60)" -ForegroundColor Gray
Write-Host "     - New methods (Part 1, lines 63-350)" -ForegroundColor Gray
Write-Host ""
Write-Host "  2. Open: backend\routes\ai.py" -ForegroundColor White
Write-Host "     Copy API routes from: backend\services\ai\shadow_model_integration.py" -ForegroundColor White
Write-Host "     - Import (Part 2, lines 356-358)" -ForegroundColor Gray
Write-Host "     - All 6 endpoints (Part 2, lines 361-520)" -ForegroundColor Gray
Write-Host ""
Write-Host "  3. Enable shadow models:" -ForegroundColor White
Write-Host "     Edit .env: ENABLE_SHADOW_MODELS=true" -ForegroundColor Gray
Write-Host ""

# Phase 5: Testing
if ($TestMode) {
    Write-Host "`n[Phase 5] Running tests..." -ForegroundColor Yellow
    
    pytest backend\tests\test_shadow_model_manager.py -v
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n  All tests passed!" -ForegroundColor Green
    } else {
        Write-Host "`n  Tests failed! Check output above." -ForegroundColor Red
        exit 1
    }
}

# Phase 6: Summary
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "   DEPLOYMENT STATUS" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Pre-flight checks:  PASSED" -ForegroundColor Green
Write-Host "  Backups:           " -NoNewline
if (-not $SkipBackup) {
    Write-Host "CREATED" -ForegroundColor Green
} else {
    Write-Host "SKIPPED" -ForegroundColor Yellow
}
Write-Host "  Environment:        CONFIGURED" -ForegroundColor Green
Write-Host "  Integration:        MANUAL STEPS REQUIRED" -ForegroundColor Yellow
Write-Host ""
Write-Host "  Next Steps:" -ForegroundColor Cyan
Write-Host "  1. Follow manual integration steps above" -ForegroundColor White
Write-Host "  2. Run tests: pytest backend\tests\test_shadow_model_manager.py" -ForegroundColor White
Write-Host "  3. Enable: Set ENABLE_SHADOW_MODELS=true in .env" -ForegroundColor White
Write-Host "  4. Monitor: python scripts\shadow_dashboard.py" -ForegroundColor White
Write-Host ""
Write-Host "  Full guide: SHADOW_MODELS_DEPLOYMENT_CHECKLIST.md" -ForegroundColor Gray
Write-Host ""
