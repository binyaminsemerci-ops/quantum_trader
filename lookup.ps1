#!/usr/bin/env pwsh
# QUANTUM TRADER - Quick System Lookup Tool
# Fast access to file locations, container info, and deployment status

param(
    [string]$Query,
    [switch]$Files,
    [switch]$Containers,
    [switch]$Streams,
    [switch]$Help
)

$InventoryFile = "$PSScriptRoot\SYSTEM_INVENTORY.yaml"

function Show-Help {
    Write-Host @"
┌─────────────────────────────────────────────────────────────────┐
│ QUANTUM TRADER - System Lookup Tool                            │
└─────────────────────────────────────────────────────────────────┘

USAGE:
  .\lookup.ps1 -Query <search_term>
  .\lookup.ps1 -Files                     # List all file locations
  .\lookup.ps1 -Containers                # List all containers
  .\lookup.ps1 -Streams                   # List Redis streams

EXAMPLES:
  .\lookup.ps1 -Query "exitbrain"         # Find exitbrain files
  .\lookup.ps1 -Query "adaptive"          # Find adaptive leverage files
  .\lookup.ps1 -Query "trade.py"          # Find specific file
  .\lookup.ps1 -Containers                # Show all containers

QUICK CHECKS:
  # What's in AI Engine container?
  docker exec quantum_ai_engine ls /app/microservices/
  
  # Check adaptive leverage status
  curl -s http://localhost:8001/health | python3 -m json.tool | grep adaptive_leverage
  
  # Redis stream lengths
  docker exec quantum_redis redis-cli XLEN quantum:stream:exchange.raw

"@
}

function Search-Inventory {
    param([string]$SearchTerm)
    
    if (-not (Test-Path $InventoryFile)) {
        Write-Host "❌ Inventory file not found: $InventoryFile" -ForegroundColor Red
        return
    }
    
    Write-Host "🔍 Searching for: $SearchTerm" -ForegroundColor Cyan
    Write-Host "─" * 70
    
    $content = Get-Content $InventoryFile -Raw
    $lines = $content -split "`n"
    
    $matches = @()
    $context = @()
    
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match $SearchTerm) {
            # Capture context (5 lines before and after)
            $start = [Math]::Max(0, $i - 5)
            $end = [Math]::Min($lines.Count - 1, $i + 5)
            
            $matchContext = @{
                Line = $i + 1
                Match = $lines[$i].Trim()
                Context = $lines[$start..$end] -join "`n"
            }
            $matches += $matchContext
        }
    }
    
    if ($matches.Count -eq 0) {
        Write-Host "❌ No matches found for: $SearchTerm" -ForegroundColor Red
        Write-Host ""
        Write-Host "💡 Try these searches:" -ForegroundColor Yellow
        Write-Host "  - exitbrain"
        Write-Host "  - adaptive"
        Write-Host "  - cross_exchange"
        Write-Host "  - redis"
        return
    }
    
    Write-Host "✅ Found $($matches.Count) matches:" -ForegroundColor Green
    Write-Host ""
    
    foreach ($match in $matches) {
        Write-Host "📍 Line $($match.Line):" -ForegroundColor Yellow
        Write-Host $match.Context
        Write-Host "─" * 70
    }
}

function Show-Files {
    Write-Host @"
┌─────────────────────────────────────────────────────────────────┐
│ FILE LOCATIONS                                                  │
└─────────────────────────────────────────────────────────────────┘

📦 PHASE 4M - Cross-Exchange Intelligence
  Location: microservices/data_collector/
  In Containers: quantum_cross_exchange, quantum_ai_engine
  Files:
    ✅ exchange_data_collector.py (385 lines)
    ✅ exchange_stream_bridge.py (270 lines) - RUNNING
    ✅ Dockerfile

  Aggregator: microservices/ai_engine/cross_exchange_aggregator.py (240 lines)
  Features: microservices/ai_engine/features/
    ✅ exchange_feature_adapter.py (220 lines)
    ✅ feature_loader.py (120 lines)

📦 PHASE 4N - Adaptive Leverage Engine
  Location: microservices/exitbrain_v3_5/
  In Containers: quantum_ai_engine
  Status: ✅ ACTIVE
  Files:
    ✅ adaptive_leverage_engine.py (305 lines)
    ✅ pnl_tracker.py (94 lines)
    ✅ __init__.py

  Integration: backend/domains/exits/exit_brain_v3/v35_integration.py (200 lines)
  Import: from backend.domains.exits.exit_brain_v3.v35_integration import get_v35_integration

📦 AI Engine Core
  Location: microservices/ai_engine/
  In Containers: quantum_ai_engine
  Files:
    ✅ service.py (1163 lines) - Core orchestration
    ✅ main.py (231 lines) - FastAPI app

📦 Backend Models
  Location: backend/models/
  In Containers: quantum_ai_engine
  Files:
    ✅ ai_training.py - FIXED (try/except import)
    ✅ trade_logs.py
    ✅ events.py
    ✅ policy.py

"@
}

function Show-Containers {
    Write-Host @"
┌─────────────────────────────────────────────────────────────────┐
│ CONTAINER STATUS                                                │
└─────────────────────────────────────────────────────────────────┘

"@
    
    # Check if on VPS or local
    if (Test-Path "~/.ssh/hetzner_fresh") {
        Write-Host "📡 Checking VPS containers..." -ForegroundColor Cyan
        ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 'docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"'
    } else {
        Write-Host "📡 Checking local containers..." -ForegroundColor Cyan
        docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    }
    
    Write-Host ""
    Write-Host "🔧 Container Details:" -ForegroundColor Yellow
    Write-Host @"
  
  quantum_redis
    Image: redis:7-alpine
    Port: 6379
    Purpose: EventBus & Cache
    Health: redis-cli ping
  
  quantum_cross_exchange
    Image: quantum_trader-cross-exchange
    Purpose: Multi-exchange WebSocket data
    Streams: quantum:stream:exchange.raw (3,897+ entries)
  
  quantum_ai_engine
    Image: quantum_trader-ai-engine
    Port: 8001
    Purpose: ML inference, adaptive leverage
    Modules: exitbrain_v3_5, cross_exchange

"@
}

function Show-Streams {
    Write-Host @"
┌─────────────────────────────────────────────────────────────────┐
│ REDIS STREAMS                                                   │
└─────────────────────────────────────────────────────────────────┘

"@
    
    if (Test-Path "~/.ssh/hetzner_fresh") {
        Write-Host "📊 Checking VPS Redis streams..." -ForegroundColor Cyan
        Write-Host ""
        
        Write-Host "quantum:stream:exchange.raw:" -ForegroundColor Yellow
        $rawLen = ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 'docker exec quantum_redis redis-cli XLEN quantum:stream:exchange.raw'
        Write-Host "  Entries: $rawLen" -ForegroundColor Green
        
        Write-Host ""
        Write-Host "quantum:stream:exchange.normalized:" -ForegroundColor Yellow
        $normLen = ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 'docker exec quantum_redis redis-cli XLEN quantum:stream:exchange.normalized'
        Write-Host "  Entries: $normLen" -ForegroundColor $(if ($normLen -eq "0") { "Red" } else { "Green" })
        
        Write-Host ""
        Write-Host "quantum:stream:exitbrain.pnl:" -ForegroundColor Yellow
        $pnlLen = ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 'docker exec quantum_redis redis-cli XLEN quantum:stream:exitbrain.pnl'
        Write-Host "  Entries: $pnlLen" -ForegroundColor $(if ($pnlLen -eq "0") { "Yellow" } else { "Green" })
    } else {
        Write-Host "⚠️  Not connected to VPS. Run on VPS for live data." -ForegroundColor Yellow
    }
    
    Write-Host ""
    Write-Host "📖 Stream Details:" -ForegroundColor Cyan
    Write-Host @"
  
  quantum:stream:exchange.raw
    Producer: exchange_stream_bridge.py
    Consumers: cross_exchange_aggregator.py
    Rate: ~200 ticks/minute
    Format: {exchange, symbol, timestamp, o, h, l, c, v}
  
  quantum:stream:exchange.normalized
    Producer: cross_exchange_aggregator.py (not running)
    Consumers: AI Engine ensemble models
    Status: Empty (aggregator needs AI Engine)
    Format: {symbol, timestamp, avg_price, price_divergence, num_exchanges}
  
  quantum:stream:exitbrain.pnl
    Producer: ExitBrain v3.5
    Consumers: PnL tracker, analytics
    Status: Ready (no trades closed yet)
    Format: {symbol, leverage, tp1-3, sl, LSF, adjustment}

"@
}

# Main execution
if ($Help -or (-not $Query -and -not $Files -and -not $Containers -and -not $Streams)) {
    Show-Help
    exit 0
}

if ($Files) {
    Show-Files
    exit 0
}

if ($Containers) {
    Show-Containers
    exit 0
}

if ($Streams) {
    Show-Streams
    exit 0
}

if ($Query) {
    Search-Inventory -SearchTerm $Query
    exit 0
}
