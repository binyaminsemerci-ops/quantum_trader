<#
.SYNOPSIS
    Validation script for Phase 4P - Adaptive Exposure Balancer
.DESCRIPTION
    Tests all Phase 4P components:
    - Exposure Balancer core module
    - Background service
    - Docker configuration
    - Health endpoint integration
.NOTES
    Run from repository root: .\scripts\validate_phase_4p.ps1
#>

param(
    [switch]$SkipVPS,
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "PHASE 4P VALIDATION - ADAPTIVE EXPOSURE BALANCER" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

$testResults = @()
$testCount = 0
$passCount = 0

function Test-Component {
    param(
        [string]$Name,
        [scriptblock]$Test,
        [string]$Category = "General"
    )
    
    $script:testCount++
    Write-Host "[$script:testCount] Testing: $Name..." -NoNewline
    
    try {
        $result = & $Test
        if ($result) {
            Write-Host " ✓ PASS" -ForegroundColor Green
            $script:passCount++
            $script:testResults += [PSCustomObject]@{
                Test = $Name
                Category = $Category
                Result = "PASS"
                Message = ""
            }
            return $true
        } else {
            Write-Host " ✗ FAIL" -ForegroundColor Red
            $script:testResults += [PSCustomObject]@{
                Test = $Name
                Category = $Category
                Result = "FAIL"
                Message = "Test returned false"
            }
            return $false
        }
    }
    catch {
        Write-Host " ✗ ERROR" -ForegroundColor Red
        if ($Verbose) {
            Write-Host "  Error: $($_.Exception.Message)" -ForegroundColor Yellow
        }
        $script:testResults += [PSCustomObject]@{
            Test = $Name
            Category = $Category
            Result = "ERROR"
            Message = $_.Exception.Message
        }
        return $false
    }
}

Write-Host "Category: Core Module" -ForegroundColor Yellow
Write-Host "---------------------------------------------" -ForegroundColor Gray

Test-Component -Name "exposure_balancer.py exists" -Category "Core Module" -Test {
    Test-Path "microservices\exposure_balancer\exposure_balancer.py"
}

Test-Component -Name "ExposureBalancer class defined" -Category "Core Module" -Test {
    $content = Get-Content "microservices\exposure_balancer\exposure_balancer.py" -Raw
    $content -match "class ExposureBalancer"
}

Test-Component -Name "Risk assessment methods present" -Category "Core Module" -Test {
    $content = Get-Content "microservices\exposure_balancer\exposure_balancer.py" -Raw
    ($content -match "def assess_risk") -and ($content -match "def execute_action")
}

Test-Component -Name "Priority-based action system" -Category "Core Module" -Test {
    $content = Get-Content "microservices\exposure_balancer\exposure_balancer.py" -Raw
    ($content -match "priority: int") -and ($content -match "priority == 1")
}

Test-Component -Name "Redis integration configured" -Category "Core Module" -Test {
    $content = Get-Content "microservices\exposure_balancer\exposure_balancer.py" -Raw
    ($content -match "quantum:stream:exposure.alerts") -and ($content -match "quantum:stream:executor.commands")
}

Write-Host ""
Write-Host "Category: Docker Setup" -ForegroundColor Yellow
Write-Host "---------------------------------------------" -ForegroundColor Gray

Test-Component -Name "Dockerfile exists" -Category "Docker" -Test {
    Test-Path "microservices\exposure_balancer\Dockerfile"
}

Test-Component -Name "service.py exists" -Category "Docker" -Test {
    Test-Path "microservices\exposure_balancer\service.py"
}

Test-Component -Name "Background service loop implemented" -Category "Docker" -Test {
    $content = Get-Content "microservices\exposure_balancer\service.py" -Raw
    ($content -match "def run_loop") -and ($content -match "rebalance\(\)")
}

Test-Component -Name "__init__.py present" -Category "Docker" -Test {
    Test-Path "microservices\exposure_balancer\__init__.py"
}

Write-Host ""
Write-Host "Category: Integration" -ForegroundColor Yellow
Write-Host "---------------------------------------------" -ForegroundColor Gray

Test-Component -Name "AI Engine health endpoint updated" -Category "Integration" -Test {
    $content = Get-Content "microservices\ai_engine\service.py" -Raw
    $content -match "exposure_balancer"
}

Test-Component -Name "docker-compose.vps.yml updated" -Category "Integration" -Test {
    $content = Get-Content "docker-compose.vps.yml" -Raw
    ($content -match "exposure-balancer:") -and ($content -match "EXPOSURE_BALANCER_ENABLED")
}

Test-Component -Name "Environment variables configured" -Category "Integration" -Test {
    $content = Get-Content "docker-compose.vps.yml" -Raw
    ($content -match "MAX_MARGIN_UTIL") -and ($content -match "REBALANCE_INTERVAL")
}

Write-Host ""
Write-Host "Category: Phase Integration" -ForegroundColor Yellow
Write-Host "---------------------------------------------" -ForegroundColor Gray

Test-Component -Name "Phase 4M+ integration (divergence)" -Category "Integration" -Test {
    $content = Get-Content "microservices\exposure_balancer\exposure_balancer.py" -Raw
    $content -match "quantum:cross:divergence"
}

Test-Component -Name "Phase 4O+ integration (confidence)" -Category "Integration" -Test {
    $content = Get-Content "microservices\exposure_balancer\exposure_balancer.py" -Raw
    $content -match "quantum:meta:confidence"
}

Test-Component -Name "Auto executor command interface" -Category "Integration" -Test {
    $content = Get-Content "microservices\exposure_balancer\exposure_balancer.py" -Raw
    $content -match "quantum:stream:executor.commands"
}

Write-Host ""
Write-Host "Category: Risk Assessment Logic" -ForegroundColor Yellow
Write-Host "---------------------------------------------" -ForegroundColor Gray

Test-Component -Name "Margin overload check (priority 1)" -Category "Logic" -Test {
    $content = Get-Content "microservices\exposure_balancer\exposure_balancer.py" -Raw
    ($content -match "margin_utilization > self.max_margin_util") -and ($content -match 'priority=1')
}

Test-Component -Name "Symbol overexposure check (priority 2)" -Category "Logic" -Test {
    $content = Get-Content "microservices\exposure_balancer\exposure_balancer.py" -Raw
    ($content -match "exposure \> self\.max_symbol_exposure") -and ($content -match 'priority=2')
}

Test-Component -Name "Diversification check" -Category "Logic" -Test {
    $content = Get-Content "microservices\exposure_balancer\exposure_balancer.py" -Raw
    $content -match "symbol_count \< self\.min_diversification"
}

Test-Component -Name "Divergence check" -Category "Logic" -Test {
    $content = Get-Content "microservices\exposure_balancer\exposure_balancer.py" -Raw
    $content -match "cross_divergence \> self\.divergence_threshold"
}

Test-Component -Name "Alert system implemented" -Category "Logic" -Test {
    $content = Get-Content "microservices\exposure_balancer\exposure_balancer.py" -Raw
    ($content -match "def _send_alert") -and ($content -match "quantum:stream:exposure.alerts")
}

# VPS Tests (optional)
if (-not $SkipVPS) {
    Write-Host ""
    Write-Host "Category: VPS Deployment" -ForegroundColor Yellow
    Write-Host "---------------------------------------------" -ForegroundColor Gray
    
    Test-Component -Name "VPS health endpoint reachable" -Category "VPS" -Test {
        try {
            $response = Invoke-RestMethod -Uri "http://46.224.116.254:8001/health" -TimeoutSec 5
            $response.exposure_balancer_enabled -ne $null
        } catch {
            $false
        }
    }
}

# Summary
Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "VALIDATION SUMMARY" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Total Tests: $testCount" -ForegroundColor White
Write-Host "Passed:      $passCount" -ForegroundColor Green
Write-Host "Failed:      $($testCount - $passCount)" -ForegroundColor $(if ($passCount -eq $testCount) { "Green" } else { "Red" })
Write-Host ""

$successRate = [math]::Round(($passCount / $testCount) * 100, 1)
Write-Host "Success Rate: $successRate%" -ForegroundColor $(if ($successRate -eq 100) { "Green" } elseif ($successRate -ge 80) { "Yellow" } else { "Red" })
Write-Host ""

# Category breakdown
$categories = $testResults | Group-Object Category
Write-Host "Results by Category:" -ForegroundColor White
foreach ($cat in $categories) {
    $catPass = ($cat.Group | Where-Object { $_.Result -eq "PASS" }).Count
    $catTotal = $cat.Count
    $catRate = [math]::Round(($catPass / $catTotal) * 100, 1)
    $color = if ($catRate -eq 100) { "Green" } elseif ($catRate -ge 80) { "Yellow" } else { "Red" }
    $percentSign = [char]37
    Write-Host ("  {0}: {1}/{2} ({3}{4})" -f $cat.Name, $catPass, $catTotal, $catRate, $percentSign) -ForegroundColor $color
}

Write-Host ""

# Failed tests details
$failed = $testResults | Where-Object { $_.Result -ne "PASS" }
if ($failed) {
    Write-Host "Failed Tests:" -ForegroundColor Red
    foreach ($test in $failed) {
        Write-Host "  - [$($test.Category)] $($test.Test)" -ForegroundColor Red
        if ($test.Message) {
            Write-Host "    $($test.Message)" -ForegroundColor Yellow
        }
    }
    Write-Host ""
}

# Final verdict
if ($passCount -eq $testCount) {
    Write-Host "✓ ALL TESTS PASSED - Phase 4P Ready for Deployment!" -ForegroundColor Green
    exit 0
} else {
    Write-Host "✗ SOME TESTS FAILED - Review above results" -ForegroundColor Red
    exit 1
}
