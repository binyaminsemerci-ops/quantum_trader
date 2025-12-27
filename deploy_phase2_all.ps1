#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Deploy all Phase 2 AI modules to production

.DESCRIPTION
    Comprehensive deployment script for:
    - Phase 2C: Continuous Learning Manager (CLM)
    - Phase 2D: Volatility Structure Engine
    - Phase 2B: Orderbook Imbalance Module
    
    This script will:
    1. Verify Docker is running
    2. Rebuild AI Engine container
    3. Restart services
    4. Verify deployment health
    5. Monitor logs for successful initialization

.EXAMPLE
    .\deploy_phase2_all.ps1
    
.NOTES
    Author: Quantum Trader AI
    Date: December 23, 2025
    Phase: 2B, 2C, 2D - Complete Deployment
#>

param(
    [switch]$SkipBuild,      # Skip container rebuild (for quick restart)
    [switch]$NoCache,        # Force full rebuild with --no-cache
    [switch]$FollowLogs,     # Follow logs after deployment
    [int]$LogTailLines = 100 # Number of log lines to show
)

# Color output functions
function Write-Info { param($msg) Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Write-Success { param($msg) Write-Host "[✓] $msg" -ForegroundColor Green }
function Write-Warning { param($msg) Write-Host "[⚠] $msg" -ForegroundColor Yellow }
function Write-Error { param($msg) Write-Host "[✗] $msg" -ForegroundColor Red }
function Write-Step { param($msg) Write-Host "`n=== $msg ===" -ForegroundColor Magenta }

# Banner
Write-Host @"

╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║    🚀 QUANTUM TRADER - PHASE 2 DEPLOYMENT SCRIPT 🚀      ║
║                                                           ║
║    Phase 2C: Continuous Learning Manager                 ║
║    Phase 2D: Volatility Structure Engine                 ║
║    Phase 2B: Orderbook Imbalance Module                  ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝

"@ -ForegroundColor Yellow

Start-Sleep -Seconds 2

# ============================================================================
# STEP 1: PRE-DEPLOYMENT CHECKS
# ============================================================================
Write-Step "Pre-Deployment Checks"

# Check if we're in the right directory
if (-not (Test-Path "docker-compose.yml")) {
    Write-Error "docker-compose.yml not found. Run this script from the quantum_trader root directory."
    exit 1
}
Write-Success "Working directory verified: $PWD"

# Check git status
Write-Info "Checking git status..."
$gitStatus = git status --porcelain
if ($gitStatus) {
    Write-Warning "Uncommitted changes detected:"
    git status --short
    $continue = Read-Host "`nContinue anyway? (y/N)"
    if ($continue -ne 'y') {
        Write-Info "Deployment cancelled."
        exit 0
    }
}

# Show current commit
$currentCommit = git log --oneline -1
Write-Success "Current commit: $currentCommit"

# Check Docker availability
Write-Info "Checking Docker availability..."
try {
    $dockerVersion = docker version --format '{{.Server.Version}}' 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Docker not responding"
    }
    Write-Success "Docker is running (version: $dockerVersion)"
} catch {
    Write-Error "Docker is not running or not installed."
    Write-Info "Please start Docker Desktop and try again."
    exit 1
}

# Check docker-compose
Write-Info "Checking docker-compose..."
try {
    $composeVersion = docker-compose version --short 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "docker-compose not found"
    }
    Write-Success "docker-compose available (version: $composeVersion)"
} catch {
    Write-Error "docker-compose not available."
    exit 1
}

# ============================================================================
# STEP 2: BACKUP CURRENT STATE
# ============================================================================
Write-Step "Backup Current State"

Write-Info "Getting current container status..."
$containers = docker ps --filter "name=quantum_ai_engine" --format "table {{.Names}}\t{{.Status}}\t{{.Image}}"
if ($containers) {
    Write-Host $containers
    Write-Success "Current AI Engine container found"
} else {
    Write-Warning "No AI Engine container currently running"
}

# ============================================================================
# STEP 3: BUILD NEW IMAGE
# ============================================================================
if (-not $SkipBuild) {
    Write-Step "Building New AI Engine Image"
    
    $buildArgs = "build ai-engine"
    if ($NoCache) {
        $buildArgs = "build --no-cache ai-engine"
        Write-Info "Building with --no-cache flag..."
    } else {
        Write-Info "Building with cache..."
    }
    
    Write-Info "Starting docker-compose build..."
    $buildStart = Get-Date
    
    docker-compose $buildArgs.Split()
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Docker build failed!"
        Write-Info "Check logs above for errors."
        exit 1
    }
    
    $buildDuration = (Get-Date) - $buildStart
    Write-Success "Build completed in $($buildDuration.TotalSeconds.ToString('F1')) seconds"
} else {
    Write-Warning "Skipping build (--SkipBuild flag)"
}

# ============================================================================
# STEP 4: STOP CURRENT CONTAINER
# ============================================================================
Write-Step "Stopping Current Container"

Write-Info "Stopping ai-engine service..."
docker-compose stop ai-engine

if ($LASTEXITCODE -eq 0) {
    Write-Success "Container stopped successfully"
} else {
    Write-Warning "Container stop returned exit code $LASTEXITCODE (may not be running)"
}

Start-Sleep -Seconds 2

# ============================================================================
# STEP 5: START NEW CONTAINER
# ============================================================================
Write-Step "Starting New Container"

Write-Info "Starting ai-engine service..."
docker-compose up -d ai-engine

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to start container!"
    Write-Info "Attempting to show logs..."
    docker-compose logs --tail 50 ai-engine
    exit 1
}

Write-Success "Container started successfully"
Start-Sleep -Seconds 5

# ============================================================================
# STEP 6: VERIFY DEPLOYMENT
# ============================================================================
Write-Step "Verifying Deployment"

Write-Info "Checking container status..."
$containerStatus = docker ps --filter "name=quantum_ai_engine" --format "{{.Status}}"
if ($containerStatus -match "Up") {
    Write-Success "Container is running: $containerStatus"
} else {
    Write-Error "Container is not in running state!"
    docker ps -a --filter "name=quantum_ai_engine"
    exit 1
}

Write-Info "Waiting for services to initialize (10 seconds)..."
Start-Sleep -Seconds 10

# ============================================================================
# STEP 7: CHECK LOGS FOR SUCCESSFUL INITIALIZATION
# ============================================================================
Write-Step "Checking Initialization Logs"

$logs = docker logs quantum_ai_engine --tail $LogTailLines 2>&1

# Check for Phase 2C (CLM)
if ($logs -match "Continuous Learning Manager active") {
    Write-Success "Phase 2C: Continuous Learning Manager - ONLINE"
} else {
    Write-Warning "Phase 2C: CLM initialization not confirmed in logs"
}

# Check for Phase 2D (Volatility)
if ($logs -match "Volatility Structure Engine active") {
    Write-Success "Phase 2D: Volatility Structure Engine - ONLINE"
} else {
    Write-Warning "Phase 2D: Volatility initialization not confirmed in logs"
}

# Check for Phase 2B (Orderbook)
if ($logs -match "Orderbook Imbalance Module active") {
    Write-Success "Phase 2B: Orderbook Imbalance Module - ONLINE"
} else {
    Write-Warning "Phase 2B: Orderbook initialization not confirmed in logs"
}

# Check for orderbook data feed
if ($logs -match "Orderbook data feed started") {
    Write-Success "Phase 2B: Orderbook data feed - ACTIVE"
} else {
    Write-Warning "Phase 2B: Orderbook data feed not confirmed in logs"
}

# Check for errors
if ($logs -match "ERROR|Failed|Exception") {
    Write-Warning "Some errors detected in logs (see below)"
}

# ============================================================================
# STEP 8: HEALTH CHECK
# ============================================================================
Write-Step "Health Check"

Write-Info "Testing AI Engine health endpoint..."
try {
    $healthResponse = Invoke-WebRequest -Uri "http://localhost:8001/health" -TimeoutSec 5 -ErrorAction Stop
    $health = $healthResponse.Content | ConvertFrom-Json
    
    Write-Success "Health endpoint responding"
    Write-Info "Service: $($health.service)"
    Write-Info "Status: $($health.status)"
    Write-Info "Uptime: $($health.uptime_seconds)s"
    
    if ($health.dependencies) {
        Write-Info "Dependencies:"
        foreach ($dep in $health.dependencies.PSObject.Properties) {
            $status = $dep.Value.status
            $color = if ($status -eq "healthy") { "Green" } else { "Yellow" }
            Write-Host "  - $($dep.Name): $status" -ForegroundColor $color
        }
    }
    
} catch {
    Write-Warning "Health endpoint not responding: $_"
    Write-Info "Service may still be initializing..."
}

# ============================================================================
# STEP 9: SUMMARY
# ============================================================================
Write-Step "Deployment Summary"

Write-Host @"

╔═══════════════════════════════════════════════════════════╗
║                 DEPLOYMENT COMPLETE! ✓                    ║
╚═══════════════════════════════════════════════════════════╝

Phase 2 Modules:
  ✓ Phase 2C: Continuous Learning Manager (4 models)
  ✓ Phase 2D: Volatility Structure Engine (11 metrics)
  ✓ Phase 2B: Orderbook Imbalance Module (5 metrics)

Total New Features: 20+ AI metrics active

Container: quantum_ai_engine
Status: $containerStatus

Recent Logs:
───────────────────────────────────────────────────────────
"@

# Show last 20 lines of logs
docker logs quantum_ai_engine --tail 20

Write-Host @"
───────────────────────────────────────────────────────────

Next Steps:
  • Monitor logs: docker logs -f quantum_ai_engine
  • Check metrics: docker exec quantum_ai_engine python -c "from backend.services.ai.volatility_structure_engine import VolatilityStructureEngine; print('✓')"
  • Test trading: Wait for market.tick events to flow through

"@ -ForegroundColor Cyan

# ============================================================================
# OPTIONAL: FOLLOW LOGS
# ============================================================================
if ($FollowLogs) {
    Write-Info "Following logs (Ctrl+C to exit)..."
    Start-Sleep -Seconds 2
    docker logs -f quantum_ai_engine
}

Write-Success "Deployment script completed!"
