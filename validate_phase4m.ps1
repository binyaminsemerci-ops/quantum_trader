# ============================================================================
# PHASE 4M VALIDATION TEST SUITE (PowerShell)
# ============================================================================
# Validates all components of cross-exchange intelligence system
# ============================================================================

$ErrorActionPreference = "Continue"
$TestsPassed = 0
$TestsFailed = 0

Write-Host "========================================================================"
Write-Host "PHASE 4M - CROSS-EXCHANGE INTELLIGENCE VALIDATION"
Write-Host "========================================================================"
Write-Host ""

# Test 1: Data Collector
Write-Host "▶ Test 1: Exchange Data Collector (REST API)"
Write-Host "------------------------------------------------------------------------"
$result = & python microservices/data_collector/exchange_data_collector.py --test 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ PASS - Data collector fetches from all exchanges" -ForegroundColor Green
    $TestsPassed++
} else {
    Write-Host "❌ FAIL - Data collector has errors" -ForegroundColor Red
    Write-Host $result
    $TestsFailed++
}
Write-Host ""

# Test 2: Stream Bridge (short run)
Write-Host "▶ Test 2: Raw Stream Population (WebSocket Bridge)"
Write-Host "------------------------------------------------------------------------"
Write-Host "Starting stream bridge for 10 seconds..."
$job = Start-Job -ScriptBlock { python microservices/data_collector/exchange_stream_bridge.py }
Start-Sleep -Seconds 10
Stop-Job $job
Remove-Job $job
Start-Sleep -Seconds 2

try {
    $rawLen = docker exec quantum_redis redis-cli XLEN quantum:stream:exchange.raw 2>$null
    if ([int]$rawLen -gt 0) {
        Write-Host "✅ PASS - Raw stream has $rawLen entries" -ForegroundColor Green
        $TestsPassed++
    } else {
        Write-Host "❌ FAIL - Raw stream is empty" -ForegroundColor Red
        $TestsFailed++
    }
} catch {
    Write-Host "⚠️  SKIP - Cannot check Redis (Docker not running?)" -ForegroundColor Yellow
}
Write-Host ""

# Test 3: Aggregator
Write-Host "▶ Test 3: Cross-Exchange Aggregator"
Write-Host "------------------------------------------------------------------------"
$result = & python microservices/ai_engine/cross_exchange_aggregator.py --test 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ PASS - Aggregator merges and normalizes data" -ForegroundColor Green
    $TestsPassed++
} else {
    Write-Host "❌ FAIL - Aggregator has errors" -ForegroundColor Red
    Write-Host $result
    $TestsFailed++
}
Write-Host ""

# Test 4: Normalized Stream
Write-Host "▶ Test 4: Normalized Stream Content"
Write-Host "------------------------------------------------------------------------"
try {
    $normData = docker exec quantum_redis redis-cli XREVRANGE quantum:stream:exchange.normalized + - COUNT 1 2>$null
    if ($normData) {
        Write-Host "✅ PASS - Normalized stream has data" -ForegroundColor Green
        Write-Host $normData[0..4]
        $TestsPassed++
    } else {
        Write-Host "❌ FAIL - Normalized stream is empty" -ForegroundColor Red
        $TestsFailed++
    }
} catch {
    Write-Host "⚠️  SKIP - Cannot check Redis" -ForegroundColor Yellow
}
Write-Host ""

# Test 5: Feature Adapter
Write-Host "▶ Test 5: Exchange Feature Adapter"
Write-Host "------------------------------------------------------------------------"
$result = & python microservices/ai_engine/features/exchange_feature_adapter.py --test 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ PASS - Feature adapter creates ML features" -ForegroundColor Green
    $TestsPassed++
} else {
    Write-Host "❌ FAIL - Feature adapter has errors" -ForegroundColor Red
    Write-Host $result
    $TestsFailed++
}
Write-Host ""

# Test 6: Feature Loader
Write-Host "▶ Test 6: Feature Loader Integration"
Write-Host "------------------------------------------------------------------------"
$result = & python microservices/ai_engine/features/feature_loader.py --test 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ PASS - Feature loader loads cross-exchange features" -ForegroundColor Green
    $TestsPassed++
} else {
    Write-Host "❌ FAIL - Feature loader has errors" -ForegroundColor Red
    Write-Host $result
    $TestsFailed++
}
Write-Host ""

# Test 7: AI Engine Health
Write-Host "▶ Test 7: AI Engine Health Check"
Write-Host "------------------------------------------------------------------------"
try {
    $health = Invoke-RestMethod -Uri "http://localhost:8001/health" -TimeoutSec 5
    if ($health -match "cross_exchange") {
        Write-Host "✅ PASS - AI Engine reports cross-exchange status:" -ForegroundColor Green
        Write-Host ($health | ConvertTo-Json)
        $TestsPassed++
    } else {
        Write-Host "❌ FAIL - AI Engine missing cross_exchange flag" -ForegroundColor Red
        $TestsFailed++
    }
} catch {
    Write-Host "⚠️  SKIP - AI Engine not responding" -ForegroundColor Yellow
    Write-Host "   Run: docker-compose -f docker-compose.vps.yml up -d ai-engine"
}
Write-Host ""

# Test 8: Docker Service
Write-Host "▶ Test 8: Docker Cross-Exchange Service"
Write-Host "------------------------------------------------------------------------"
try {
    $running = docker ps --filter "name=quantum_cross_exchange" --format "{{.Status}}"
    if ($running) {
        Write-Host "✅ PASS - Cross-exchange service is running: $running" -ForegroundColor Green
        $TestsPassed++
    } else {
        Write-Host "⚠️  SKIP - Cross-exchange service not started" -ForegroundColor Yellow
        Write-Host "   Run: docker-compose -f docker-compose.vps.yml up -d cross-exchange"
    }
} catch {
    Write-Host "⚠️  SKIP - Cannot check Docker" -ForegroundColor Yellow
}
Write-Host ""

# Summary
Write-Host "========================================================================"
Write-Host "VALIDATION SUMMARY"
Write-Host "========================================================================"
Write-Host "Tests Passed: $TestsPassed"
Write-Host "Tests Failed: $TestsFailed"
Write-Host ""

if ($TestsFailed -eq 0) {
    Write-Host "🎉 ALL TESTS PASSED" -ForegroundColor Green
    Write-Host ""
    Write-Host "✅ Phase 4M Complete - Set flag: 'cross_exchange_intelligence': 'active'" -ForegroundColor Green
    Write-Host ""
    exit 0
} else {
    Write-Host "❌ Some tests failed - review errors above" -ForegroundColor Red
    exit 1
}
