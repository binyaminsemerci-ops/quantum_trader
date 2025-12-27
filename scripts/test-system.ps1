# Quantum Trader - Complete System Test Script
# PowerShell version for comprehensive testing

param(
    [Parameter(Mandatory=$false)]
    [string]$BaseUrl = "http://localhost:8000",
    
    [Parameter(Mandatory=$false)]
    [switch]$DetailedOutput
)

$ErrorActionPreference = "Continue"
$passedTests = 0
$failedTests = 0
$testResults = @()

function Write-TestHeader {
    param([string]$message)
    Write-Host "`n╔══════════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "║ $($message.PadRight(48)) ║" -ForegroundColor Cyan
    Write-Host "╚══════════════════════════════════════════════════╝" -ForegroundColor Cyan
}

function Test-Endpoint {
    param(
        [string]$name,
        [string]$url,
        [scriptblock]$validation,
        [int]$timeout = 10
    )
    
    Write-Host "`n🧪 Testing: $name" -ForegroundColor Yellow
    
    try {
        $response = Invoke-RestMethod -Uri $url -TimeoutSec $timeout -ErrorAction Stop
        
        if ($validation) {
            $result = & $validation $response
            if ($result) {
                Write-Host "   ✅ PASSED" -ForegroundColor Green
                $script:passedTests++
                $script:testResults += [PSCustomObject]@{
                    Test = $name
                    Status = "PASSED"
                    Details = "OK"
                }
                return $true
            } else {
                Write-Host "   ❌ FAILED - Validation failed" -ForegroundColor Red
                $script:failedTests++
                $script:testResults += [PSCustomObject]@{
                    Test = $name
                    Status = "FAILED"
                    Details = "Validation failed"
                }
                return $false
            }
        }
        
        Write-Host "   ✅ PASSED" -ForegroundColor Green
        $script:passedTests++
        $script:testResults += [PSCustomObject]@{
            Test = $name
            Status = "PASSED"
            Details = "OK"
        }
        return $true
        
    } catch {
        Write-Host "   ❌ FAILED - $($_.Exception.Message)" -ForegroundColor Red
        $script:failedTests++
        $script:testResults += [PSCustomObject]@{
            Test = $name
            Status = "FAILED"
            Details = $_.Exception.Message
        }
        return $false
    }
}

Write-Host @"
═══════════════════════════════════════════════════════
   🚀 QUANTUM TRADER - COMPLETE SYSTEM TEST
═══════════════════════════════════════════════════════
"@ -ForegroundColor Cyan

Write-Host "Target: $BaseUrl`n" -ForegroundColor White

# Test 1: System Health
Write-TestHeader "SYSTEM HEALTH CHECKS"

Test-Endpoint `
    -name "Backend Health" `
    -url "$BaseUrl/health" `
    -validation {
        param($data)
        return $data.status -eq "healthy"
    }

Test-Endpoint `
    -name "Scheduler Status" `
    -url "$BaseUrl/health" `
    -validation {
        param($data)
        return $data.scheduler.running -eq $true
    }

# Test 2: AI Engine
Write-TestHeader "AI ENGINE TESTS"

Test-Endpoint `
    -name "AI Model Info" `
    -url "$BaseUrl/api/ai/model/info" `
    -validation {
        param($data)
        return $data.status -eq "Ready" -and $data.accuracy -gt 0
    }

Test-Endpoint `
    -name "Signal Generation" `
    -url "$BaseUrl/api/ai/signals/latest" `
    -validation {
        param($data)
        return $data.Count -gt 0 -and $data[0].symbol -ne $null
    }

# Test 3: Market Data
Write-TestHeader "MARKET DATA TESTS"

Test-Endpoint `
    -name "Price Fetch (BTCUSDT)" `
    -url "$BaseUrl/api/prices/latest?symbol=BTCUSDT" `
    -validation {
        param($data)
        return $data.price -gt 0
    }

Test-Endpoint `
    -name "Candles Fetch" `
    -url "$BaseUrl/api/candles?symbol=BTCUSDT&limit=100" `
    -validation {
        param($data)
        return $data.candles.Count -gt 0
    }

# Test 4: Risk Management
Write-TestHeader "RISK MANAGEMENT TESTS"

Test-Endpoint `
    -name "Risk Limits" `
    -url "$BaseUrl/api/risk" `
    -validation {
        param($data)
        return $data.max_position_size -gt 0
    }

Test-Endpoint `
    -name "Position Tracking" `
    -url "$BaseUrl/api/positions"

# Test 5: Metrics and Analytics
Write-TestHeader "METRICS & ANALYTICS TESTS"

Test-Endpoint `
    -name "Trading Metrics" `
    -url "$BaseUrl/api/metrics" `
    -validation {
        param($data)
        return $data.total_trades -ne $null
    }

Test-Endpoint `
    -name "Stats Overview" `
    -url "$BaseUrl/api/stats/overview" `
    -validation {
        param($data)
        return $data.pnl -ne $null
    }

# Test 6: Performance
Write-TestHeader "PERFORMANCE TESTS"

Write-Host "`n🧪 Testing: API Response Time" -ForegroundColor Yellow
$start = Get-Date
try {
    $response = Invoke-RestMethod -Uri "$BaseUrl/health" -TimeoutSec 5
    $end = Get-Date
    $latency = ($end - $start).TotalSeconds
    
    if ($latency -lt 1.0) {
        Write-Host "   ✅ PASSED - Latency: $([math]::Round($latency * 1000))ms" -ForegroundColor Green
        $passedTests++
        $testResults += [PSCustomObject]@{
            Test = "API Response Time"
            Status = "PASSED"
            Details = "$([math]::Round($latency * 1000))ms"
        }
    } else {
        Write-Host "   ⚠️ WARNING - High latency: $([math]::Round($latency * 1000))ms" -ForegroundColor Yellow
        $passedTests++
        $testResults += [PSCustomObject]@{
            Test = "API Response Time"
            Status = "WARNING"
            Details = "$([math]::Round($latency * 1000))ms (high)"
        }
    }
} catch {
    Write-Host "   ❌ FAILED - $($_.Exception.Message)" -ForegroundColor Red
    $failedTests++
}

# Test 7: Database
Write-TestHeader "DATABASE TESTS"

Test-Endpoint `
    -name "Trades Retrieval" `
    -url "$BaseUrl/api/trades"

Test-Endpoint `
    -name "Trade Logs" `
    -url "$BaseUrl/api/trade-logs/latest?limit=10"

# Test 8: Docker Status (if applicable)
Write-TestHeader "INFRASTRUCTURE CHECKS"

Write-Host "`n🧪 Testing: Docker Containers" -ForegroundColor Yellow
try {
    $containers = docker ps --format "table {{.Names}}\t{{.Status}}" 2>$null
    if ($containers -match "quantum_backend") {
        Write-Host "   ✅ PASSED - Backend container running" -ForegroundColor Green
        $passedTests++
        $testResults += [PSCustomObject]@{
            Test = "Docker Backend"
            Status = "PASSED"
            Details = "Container running"
        }
    } else {
        Write-Host "   ⚠️ WARNING - Backend container not found" -ForegroundColor Yellow
        $testResults += [PSCustomObject]@{
            Test = "Docker Backend"
            Status = "WARNING"
            Details = "Container not found or not using Docker"
        }
    }
} catch {
    Write-Host "   ℹ️ INFO - Docker not available or not used" -ForegroundColor Gray
}

# Summary
Write-Host "`n`n" -NoNewline
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "   📊 TEST SUMMARY" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan

$totalTests = $passedTests + $failedTests
$successRate = if ($totalTests -gt 0) { ($passedTests / $totalTests) * 100 } else { 0 }

Write-Host ""
Write-Host "   Total Tests:    $totalTests" -ForegroundColor White
Write-Host "   Passed:         $passedTests" -ForegroundColor Green
Write-Host "   Failed:         $failedTests" -ForegroundColor $(if($failedTests -gt 0){"Red"}else{"Green"})
Write-Host "   Success Rate:   $([math]::Round($successRate, 1))%" -ForegroundColor $(if($successRate -ge 90){"Green"}elseif($successRate -ge 70){"Yellow"}else{"Red"})
Write-Host ""

if ($DetailedOutput) {
    Write-Host "`n═══════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "   📋 DETAILED RESULTS" -ForegroundColor Cyan
    Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
    $testResults | Format-Table -AutoSize
}

if ($failedTests -eq 0) {
    Write-Host "   🎉 ALL TESTS PASSED!" -ForegroundColor Green
    Write-Host "═══════════════════════════════════════════════════════`n" -ForegroundColor Cyan
    exit 0
} else {
    Write-Host "   ⚠️ SOME TESTS FAILED" -ForegroundColor Red
    Write-Host "═══════════════════════════════════════════════════════`n" -ForegroundColor Cyan
    exit 1
}
