# Phase 4S+ - System Integration Verification Script
# Verifies all integration points are working correctly

param(
    [string]$VpsHost = "46.224.116.254",
    [string]$VpsUser = "qt",
    [string]$SshKey = "~/.ssh/hetzner_fresh"
)

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "     🔍 PHASE 4S+ - SYSTEM INTEGRATION VERIFICATION" -ForegroundColor White
Write-Host "═══════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

$tests = @()

# Test 1: Strategic Memory Container
Write-Host "1️⃣  Testing Strategic Memory Container..." -ForegroundColor Yellow
$containerStatus = wsl ssh -i $SshKey ${VpsUser}@${VpsHost} "docker ps --filter name=quantum_strategic_memory --format '{{.Status}}'"
if ($containerStatus -match "healthy|Up") {
    Write-Host "   ✅ Container is running: $containerStatus" -ForegroundColor Green
    $tests += @{Name="Strategic Memory Container"; Status="PASS"}
} else {
    Write-Host "   ❌ Container is not healthy: $containerStatus" -ForegroundColor Red
    $tests += @{Name="Strategic Memory Container"; Status="FAIL"}
}

# Test 2: Redis Connectivity
Write-Host "2️⃣  Testing Redis Connectivity..." -ForegroundColor Yellow
$redisPing = wsl ssh -i $SshKey ${VpsUser}@${VpsHost} "docker exec quantum_redis redis-cli PING"
if ($redisPing -eq "PONG") {
    Write-Host "   ✅ Redis is reachable" -ForegroundColor Green
    $tests += @{Name="Redis Connectivity"; Status="PASS"}
} else {
    Write-Host "   ❌ Redis is not reachable" -ForegroundColor Red
    $tests += @{Name="Redis Connectivity"; Status="FAIL"}
}

# Test 3: Feedback Key Exists
Write-Host "3️⃣  Testing Feedback Key..." -ForegroundColor Yellow
$feedback = wsl ssh -i $SshKey ${VpsUser}@${VpsHost} "docker exec quantum_redis redis-cli GET quantum:feedback:strategic_memory"
if ($feedback -and $feedback -ne "(nil)") {
    Write-Host "   ✅ Feedback key exists and populated" -ForegroundColor Green
    $tests += @{Name="Feedback Key"; Status="PASS"}
} else {
    Write-Host "   ⚠️  Feedback key not yet generated (needs 3+ samples)" -ForegroundColor Yellow
    $tests += @{Name="Feedback Key"; Status="PENDING"}
}

# Test 4: AI Engine Integration
Write-Host "4️⃣  Testing AI Engine Integration..." -ForegroundColor Yellow
$healthResponse = wsl ssh -i $SshKey ${VpsUser}@${VpsHost} "curl -s http://localhost:8001/health"
if ($healthResponse) {
    try {
        $health = $healthResponse | ConvertFrom-Json
        if ($health.metrics.strategic_memory) {
            Write-Host "   ✅ AI Engine exposes strategic_memory metrics" -ForegroundColor Green
            Write-Host "      Status: $($health.metrics.strategic_memory.status)" -ForegroundColor Cyan
            $tests += @{Name="AI Engine Integration"; Status="PASS"}
        } else {
            Write-Host "   ❌ strategic_memory metrics not found" -ForegroundColor Red
            $tests += @{Name="AI Engine Integration"; Status="FAIL"}
        }
    } catch {
        Write-Host "   ❌ Could not parse AI Engine response" -ForegroundColor Red
        $tests += @{Name="AI Engine Integration"; Status="FAIL"}
    }
} else {
    Write-Host "   ❌ AI Engine not responding" -ForegroundColor Red
    $tests += @{Name="AI Engine Integration"; Status="FAIL"}
}

# Test 5: Portfolio Governance Link
Write-Host "5️⃣  Testing Portfolio Governance Link..." -ForegroundColor Yellow
$policy = wsl ssh -i $SshKey ${VpsUser}@${VpsHost} "docker exec quantum_redis redis-cli GET quantum:governance:policy"
if ($policy) {
    Write-Host "   ✅ Governance policy key exists: $policy" -ForegroundColor Green
    $tests += @{Name="Portfolio Governance"; Status="PASS"}
} else {
    Write-Host "   ⚠️  Governance policy not yet set" -ForegroundColor Yellow
    $tests += @{Name="Portfolio Governance"; Status="PENDING"}
}

# Test 6: Meta-Regime Stream
Write-Host "6️⃣  Testing Meta-Regime Stream..." -ForegroundColor Yellow
$streamLen = wsl ssh -i $SshKey ${VpsUser}@${VpsHost} "docker exec quantum_redis redis-cli XLEN quantum:stream:meta.regime"
if ([int]$streamLen -gt 0) {
    Write-Host "   ✅ Meta-regime stream has $streamLen observations" -ForegroundColor Green
    $tests += @{Name="Meta-Regime Stream"; Status="PASS"}
} else {
    Write-Host "   ⚠️  Meta-regime stream is empty" -ForegroundColor Yellow
    $tests += @{Name="Meta-Regime Stream"; Status="PENDING"}
}

# Test 7: Service Logs
Write-Host "7️⃣  Testing Service Logs..." -ForegroundColor Yellow
$logs = wsl ssh -i $SshKey ${VpsUser}@${VpsHost} "docker logs --tail 5 quantum_strategic_memory 2>&1"
if ($logs -match "Memory sync iteration complete") {
    Write-Host "   ✅ Service is actively processing iterations" -ForegroundColor Green
    $tests += @{Name="Service Logs"; Status="PASS"}
} elseif ($logs -match "error|Error|ERROR") {
    Write-Host "   ❌ Service logs contain errors" -ForegroundColor Red
    $tests += @{Name="Service Logs"; Status="FAIL"}
} else {
    Write-Host "   ⚠️  Service has not completed an iteration yet" -ForegroundColor Yellow
    $tests += @{Name="Service Logs"; Status="PENDING"}
}

# Test 8: Event Bus Integration
Write-Host "8️⃣  Testing Event Bus Integration..." -ForegroundColor Yellow
# Check if the service can publish events
$pubsubChannels = wsl ssh -i $SshKey ${VpsUser}@${VpsHost} "docker exec quantum_redis redis-cli PUBSUB CHANNELS quantum:events:*"
if ($pubsubChannels -match "strategic_feedback") {
    Write-Host "   ✅ Event bus channel exists" -ForegroundColor Green
    $tests += @{Name="Event Bus"; Status="PASS"}
} else {
    Write-Host "   ⚠️  Event bus channel not active (events may not have been published yet)" -ForegroundColor Yellow
    $tests += @{Name="Event Bus"; Status="PENDING"}
}

# Summary
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "     📊 VERIFICATION SUMMARY" -ForegroundColor White
Write-Host "═══════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

$passed = ($tests | Where-Object {$_.Status -eq "PASS"}).Count
$failed = ($tests | Where-Object {$_.Status -eq "FAIL"}).Count
$pending = ($tests | Where-Object {$_.Status -eq "PENDING"}).Count
$total = $tests.Count

foreach ($test in $tests) {
    $color = switch ($test.Status) {
        "PASS" { "Green" }
        "FAIL" { "Red" }
        "PENDING" { "Yellow" }
    }
    Write-Host "   $($test.Status.PadRight(10)) | $($test.Name)" -ForegroundColor $color
}

Write-Host ""
Write-Host "   Total Tests:    $total" -ForegroundColor White
Write-Host "   Passed:         $passed" -ForegroundColor Green
Write-Host "   Failed:         $failed" -ForegroundColor $(if($failed -gt 0) {'Red'} else {'Green'})
Write-Host "   Pending:        $pending" -ForegroundColor Yellow
Write-Host ""

if ($failed -eq 0 -and $passed -gt 5) {
    Write-Host "🎉  ALL CRITICAL TESTS PASSED!" -ForegroundColor Green
    Write-Host "   Phase 4S+ is fully operational and integrated." -ForegroundColor White
} elseif ($failed -gt 0) {
    Write-Host "⚠️  SOME TESTS FAILED!" -ForegroundColor Red
    Write-Host "   Review the failures and check container logs for details." -ForegroundColor White
} else {
    Write-Host "⏳  SYSTEM IS WARMING UP..." -ForegroundColor Yellow
    Write-Host "   Wait for data collection and retry verification." -ForegroundColor White
}

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
