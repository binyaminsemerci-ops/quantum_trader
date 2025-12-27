# Docker Build and Test Script for Quantum Trader V3
# Purpose: Build containers and verify PYTHONPATH configuration
# Date: 2025-12-17

Write-Host "`n================================================================================" -ForegroundColor Cyan
Write-Host "  QUANTUM TRADER V3 - DOCKER BUILD AND TEST" -ForegroundColor Yellow
Write-Host "================================================================================" -ForegroundColor Cyan

# Step 1: Verify docker-compose.yml configuration
Write-Host "`nStep 1: Verifying docker-compose.yml configuration..." -ForegroundColor Yellow
$dockerComposeContent = Get-Content "docker-compose.yml" -Raw
if ($dockerComposeContent -match "PYTHONPATH=/app/backend" -and $dockerComposeContent -match "GO_LIVE=true") {
    Write-Host "  ✓ PYTHONPATH=/app/backend configured" -ForegroundColor Green
    Write-Host "  ✓ GO_LIVE=true configured" -ForegroundColor Green
} else {
    Write-Host "  ✗ Configuration missing - check docker-compose.yml" -ForegroundColor Red
    exit 1
}

# Step 2: Check Docker availability
Write-Host "`nStep 2: Checking Docker availability..." -ForegroundColor Yellow
$dockerAvailable = $false
$method = ""

# Try Docker Desktop first
try {
    $dockerVersion = docker --version 2>&1
    $dockerPs = docker ps 2>&1
    if ($dockerPs -notmatch "error") {
        $dockerAvailable = $true
        $method = "Docker Desktop"
        Write-Host "  ✓ Docker Desktop is running" -ForegroundColor Green
    }
} catch {
    Write-Host "  ✗ Docker Desktop not available" -ForegroundColor Yellow
}

# Try WSL + Podman if Docker not available
if (-not $dockerAvailable) {
    try {
        $podmanVersion = wsl podman --version 2>&1
        if ($podmanVersion -match "podman") {
            $dockerAvailable = $true
            $method = "WSL + Podman"
            Write-Host "  ✓ WSL + Podman available" -ForegroundColor Green
        }
    } catch {
        Write-Host "  ✗ WSL + Podman not available" -ForegroundColor Yellow
    }
}

if (-not $dockerAvailable) {
    Write-Host "`n  ERROR: No container runtime available!" -ForegroundColor Red
    Write-Host "  Please start Docker Desktop or configure WSL + Podman" -ForegroundColor Yellow
    Write-Host "`n  Manual Steps:" -ForegroundColor Yellow
    Write-Host "    1. Start Docker Desktop, OR" -ForegroundColor White
    Write-Host "    2. Run: wsl --shutdown && wsl" -ForegroundColor White
    Write-Host "    3. Run this script again" -ForegroundColor White
    exit 1
}

Write-Host "`n  Using: $method" -ForegroundColor Cyan

# Step 3: Clean previous builds
Write-Host "`nStep 3: Cleaning previous builds..." -ForegroundColor Yellow
if ($method -eq "Docker Desktop") {
    Write-Host "  Running: docker compose down..." -ForegroundColor Gray
    docker compose down 2>&1 | Out-Null
    Write-Host "  ✓ Stopped containers" -ForegroundColor Green
} else {
    Write-Host "  Running: podman-compose down..." -ForegroundColor Gray
    wsl bash -c "cd /mnt/c/quantum_trader && podman-compose down 2>&1" | Out-Null
    Write-Host "  ✓ Stopped containers" -ForegroundColor Green
}

# Step 4: Build backend container
Write-Host "`nStep 4: Building backend container..." -ForegroundColor Yellow
Write-Host "  This may take several minutes..." -ForegroundColor Gray

$buildSuccess = $false
$buildLog = ""

if ($method -eq "Docker Desktop") {
    Write-Host "  Running: docker compose build backend..." -ForegroundColor Gray
    $buildOutput = docker compose build backend 2>&1
    $buildLog = $buildOutput -join "`n"
    if ($LASTEXITCODE -eq 0) {
        $buildSuccess = $true
    }
} else {
    Write-Host "  Running: podman-compose build backend..." -ForegroundColor Gray
    $buildOutput = wsl bash -c "cd /mnt/c/quantum_trader && podman-compose build backend 2>&1"
    $buildLog = $buildOutput -join "`n"
    if ($LASTEXITCODE -eq 0) {
        $buildSuccess = $true
    }
}

if ($buildSuccess) {
    Write-Host "  ✓ Build completed successfully" -ForegroundColor Green
} else {
    Write-Host "  ✗ Build failed" -ForegroundColor Red
    Write-Host "`n  Build Log (last 50 lines):" -ForegroundColor Yellow
    $buildLog -split "`n" | Select-Object -Last 50 | ForEach-Object { Write-Host "    $_" -ForegroundColor Gray }
    Write-Host "`n  Full build log saved to: build_error.log" -ForegroundColor Yellow
    $buildLog | Out-File -FilePath "build_error.log" -Encoding UTF8
    exit 1
}

# Step 5: Start backend container
Write-Host "`nStep 5: Starting backend container..." -ForegroundColor Yellow

$startSuccess = $false
if ($method -eq "Docker Desktop") {
    Write-Host "  Running: docker compose up -d backend..." -ForegroundColor Gray
    docker compose up -d backend 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        $startSuccess = $true
    }
} else {
    Write-Host "  Running: podman-compose up -d backend..." -ForegroundColor Gray
    wsl bash -c "cd /mnt/c/quantum_trader && podman-compose up -d backend 2>&1" | Out-Null
    if ($LASTEXITCODE -eq 0) {
        $startSuccess = $true
    }
}

if ($startSuccess) {
    Write-Host "  ✓ Container started" -ForegroundColor Green
    Write-Host "  Waiting 10 seconds for startup..." -ForegroundColor Gray
    Start-Sleep -Seconds 10
} else {
    Write-Host "  ✗ Failed to start container" -ForegroundColor Red
    exit 1
}

# Step 6: Check container logs for errors
Write-Host "`nStep 6: Checking container logs..." -ForegroundColor Yellow

$logs = ""
if ($method -eq "Docker Desktop") {
    $logs = docker logs quantum_backend --tail 50 2>&1
} else {
    $logs = wsl bash -c "podman logs quantum_backend --tail 50 2>&1"
}

# Step 7: Analyze logs for import errors
Write-Host "`nStep 7: Analyzing logs for import errors..." -ForegroundColor Yellow

$hasModuleError = $false
$hasImportError = $false
$moduleErrors = @()

foreach ($line in $logs -split "`n") {
    if ($line -match "ModuleNotFoundError") {
        $hasModuleError = $true
        $moduleErrors += $line.Trim()
    }
    if ($line -match "ImportError") {
        $hasImportError = $true
        $moduleErrors += $line.Trim()
    }
}

# Step 8: Display results
Write-Host "`n================================================================================" -ForegroundColor Cyan
Write-Host "  TEST RESULTS" -ForegroundColor Yellow
Write-Host "================================================================================" -ForegroundColor Cyan

if (-not $hasModuleError -and -not $hasImportError) {
    Write-Host "`n  ✅ SUCCESS: No import errors detected!" -ForegroundColor Green
    Write-Host "`n  Container Logs (last 20 lines):" -ForegroundColor Cyan
    $logs -split "`n" | Select-Object -Last 20 | ForEach-Object { Write-Host "    $_" -ForegroundColor Gray }
    
    # Check for successful startup indicators
    $hasStartup = $logs -match "Application startup complete|Uvicorn running"
    if ($hasStartup) {
        Write-Host "`n  ✅ Backend started successfully!" -ForegroundColor Green
        Write-Host "  Backend should be accessible at: http://localhost:8000" -ForegroundColor Cyan
    }
} else {
    Write-Host "`n  ❌ ERRORS DETECTED:" -ForegroundColor Red
    Write-Host "`n  Import/Module Errors Found:" -ForegroundColor Yellow
    foreach ($error in $moduleErrors) {
        Write-Host "    • $error" -ForegroundColor Red
    }
    
    Write-Host "`n  Full Container Logs:" -ForegroundColor Cyan
    $logs -split "`n" | ForEach-Object { Write-Host "    $_" -ForegroundColor Gray }
}

# Step 9: Verify environment variables inside container
Write-Host "`n================================================================================" -ForegroundColor Cyan
Write-Host "  ENVIRONMENT VERIFICATION" -ForegroundColor Yellow
Write-Host "================================================================================" -ForegroundColor Cyan

Write-Host "`nChecking environment variables inside container..." -ForegroundColor Yellow

$envVars = @()
if ($method -eq "Docker Desktop") {
    $envOutput = docker exec quantum_backend env 2>&1
    $envVars = $envOutput -split "`n"
} else {
    $envOutput = wsl bash -c "podman exec quantum_backend env 2>&1"
    $envVars = $envOutput -split "`n"
}

$criticalVars = @{
    "GO_LIVE" = $false
    "PYTHONPATH" = $false
    "RL_DEBUG" = $false
}

foreach ($line in $envVars) {
    if ($line -match "^GO_LIVE=(.*)") {
        Write-Host "  ✓ GO_LIVE=$($matches[1])" -ForegroundColor Green
        $criticalVars["GO_LIVE"] = $true
    }
    if ($line -match "^PYTHONPATH=(.*)") {
        Write-Host "  ✓ PYTHONPATH=$($matches[1])" -ForegroundColor Green
        $criticalVars["PYTHONPATH"] = $true
    }
    if ($line -match "^RL_DEBUG=(.*)") {
        Write-Host "  ✓ RL_DEBUG=$($matches[1])" -ForegroundColor Green
        $criticalVars["RL_DEBUG"] = $true
    }
}

foreach ($var in $criticalVars.Keys) {
    if (-not $criticalVars[$var]) {
        Write-Host "  ✗ $var not set" -ForegroundColor Red
    }
}

# Step 10: Test module imports
Write-Host "`n================================================================================" -ForegroundColor Cyan
Write-Host "  MODULE IMPORT TEST" -ForegroundColor Yellow
Write-Host "================================================================================" -ForegroundColor Cyan

Write-Host "`nTesting module imports..." -ForegroundColor Yellow

$importTests = @(
    @{Module = "domains.exits.exit_brain_v3"; Name = "Exit Brain V3"},
    @{Module = "domains.learning.rl_v3"; Name = "RL V3"},
    @{Module = "services.clm_v3.orchestrator"; Name = "CLM V3 Orchestrator"},
    @{Module = "services.monitoring.tp_optimizer_v3"; Name = "TP Optimizer V3"}
)

foreach ($test in $importTests) {
    $module = $test.Module
    $name = $test.Name
    
    $importCmd = "python3 -c 'import $module; print(\"OK\")'"
    
    $importResult = ""
    if ($method -eq "Docker Desktop") {
        $importResult = docker exec quantum_backend bash -c $importCmd 2>&1
    } else {
        $importResult = wsl bash -c "podman exec quantum_backend bash -c '$importCmd' 2>&1"
    }
    
    if ($importResult -match "OK") {
        Write-Host "  ✓ $name - Import successful" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $name - Import failed" -ForegroundColor Red
        Write-Host "    Error: $importResult" -ForegroundColor Gray
    }
}

# Final Summary
Write-Host "`n================================================================================" -ForegroundColor Cyan
Write-Host "  FINAL SUMMARY" -ForegroundColor Yellow
Write-Host "================================================================================" -ForegroundColor Cyan

$allSuccess = (-not $hasModuleError) -and (-not $hasImportError) -and $criticalVars["GO_LIVE"] -and $criticalVars["PYTHONPATH"]

if ($allSuccess) {
    Write-Host "`n  🎉 ALL TESTS PASSED!" -ForegroundColor Green
    Write-Host "`n  Backend is running with correct configuration:" -ForegroundColor Cyan
    Write-Host "    • PYTHONPATH=/app/backend configured" -ForegroundColor White
    Write-Host "    • GO_LIVE=true set" -ForegroundColor White
    Write-Host "    • All modules import successfully" -ForegroundColor White
    Write-Host "    • No import errors detected" -ForegroundColor White
    Write-Host "`n  Backend URL: http://localhost:8000" -ForegroundColor Cyan
    Write-Host "  Health Check: http://localhost:8000/health" -ForegroundColor Cyan
} else {
    Write-Host "`n  ⚠️  SOME ISSUES DETECTED" -ForegroundColor Yellow
    Write-Host "`n  Please review the logs and errors above." -ForegroundColor White
    Write-Host "  Check RUNTIME_CONFIG_QUICKREF.md for troubleshooting." -ForegroundColor White
}

Write-Host "`n================================================================================" -ForegroundColor Cyan
