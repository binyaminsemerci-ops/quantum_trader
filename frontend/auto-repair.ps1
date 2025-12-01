# Dashboard Auto-Repair PowerShell Script
# Enkel kommando for å administrere dashboard auto-repair fra PowerShell

param(
    [Parameter(Position=0)]
    [string]$Action = "help",
    
    [Parameter(Position=1)]
    [string]$Option = ""
)

$ScriptPath = ".\src\utils\auto-repair-cli.js"

Write-Host "🤖 Quantum Trader Dashboard Auto-Repair" -ForegroundColor Cyan
Write-Host "=======================================" -ForegroundColor Cyan

# Ensure we're in frontend directory
if (!(Test-Path "package.json") -or !(Test-Path $ScriptPath)) {
    Write-Host "❌ Please run this script from the frontend directory" -ForegroundColor Red
    Write-Host "Current directory: $(Get-Location)" -ForegroundColor Yellow
    exit 1
}

switch ($Action.ToLower()) {
    "check" {
        Write-Host "🔍 Running dashboard health check..." -ForegroundColor Yellow
        & node $ScriptPath check
    }
    
    "repair" {
        Write-Host "🔧 Running automatic repair..." -ForegroundColor Yellow
        & node $ScriptPath repair
    }
    
    "reset" {
        Write-Host "🎯 Resetting to optimal layout..." -ForegroundColor Yellow
        & node $ScriptPath reset
    }
    
    "corrupt" {
        if ($Option -eq "") {
            Write-Host "❌ Please specify corruption type:" -ForegroundColor Red
            Write-Host "   candles-in-header, narrow-trade-history, missing-grid, corrupted-imports" -ForegroundColor Yellow
            exit 1
        }
        Write-Host "🧪 Simulating corruption: $Option..." -ForegroundColor Yellow
        & node $ScriptPath corrupt $Option
    }
    
    "status" {
        & node $ScriptPath status
    }
    
    "monitor" {
        Write-Host "👁️ Starting continuous monitoring (Ctrl+C to stop)..." -ForegroundColor Green
        while ($true) {
            Clear-Host
            Write-Host "🤖 Dashboard Health Monitor - $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Cyan
            Write-Host "===========================================" -ForegroundColor Cyan
            & node $ScriptPath check
            Write-Host "`n⏰ Next check in 30 seconds... (Ctrl+C to stop)" -ForegroundColor Gray
            Start-Sleep -Seconds 30
        }
    }
    
    "auto" {
        Write-Host "🚀 Starting automatic monitoring with repair..." -ForegroundColor Green
        Write-Host "This will check every 30 seconds and auto-repair critical issues" -ForegroundColor Yellow
        Write-Host "Press Ctrl+C to stop`n" -ForegroundColor Gray
        
        while ($true) {
            $checkResult = & node $ScriptPath check
            
            # Check if repair is needed (basic check by looking for issues)
            if ($checkResult -match "ISSUES DETECTED") {
                Write-Host "`n🚨 Issues detected! Running auto-repair..." -ForegroundColor Red
                & node $ScriptPath repair
                Write-Host "✅ Auto-repair completed" -ForegroundColor Green
            }
            
            Write-Host "⏰ Next check in 30 seconds..." -ForegroundColor Gray
            Start-Sleep -Seconds 30
        }
    }
    
    "help" {
        Write-Host @"

Dashboard Auto-Repair PowerShell Commands:

Basic Commands:
  .\auto-repair.ps1 check         - Check dashboard health
  .\auto-repair.ps1 repair        - Run automatic repair
  .\auto-repair.ps1 reset         - Reset to optimal layout
  .\auto-repair.ps1 status        - Show detailed status

Testing Commands:
  .\auto-repair.ps1 corrupt <type> - Simulate corruption for testing
    Available types: candles-in-header, narrow-trade-history, missing-grid, corrupted-imports

Advanced Commands:
  .\auto-repair.ps1 monitor       - Continuous health monitoring
  .\auto-repair.ps1 auto          - Automatic monitoring with repair

Examples:
  .\auto-repair.ps1 check
  .\auto-repair.ps1 corrupt narrow-trade-history
  .\auto-repair.ps1 repair
  .\auto-repair.ps1 auto

🎯 Dette systemet løser problemet med manuell reparasjon!
   Nå kan du automatisk fikse layout problemer uten å redigere kode manuelt.

"@ -ForegroundColor White
    }
    
    default {
        Write-Host "❌ Unknown action: $Action" -ForegroundColor Red
        Write-Host "Use '.\auto-repair.ps1 help' to see available commands" -ForegroundColor Yellow
    }
}