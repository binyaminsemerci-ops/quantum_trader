# Chaos Engineering Test: Redis Outage During Trading
# Tests all 5 critical infrastructure resilience fixes

param(
    [int]$OutageDurationSeconds = 60,
    [switch]$SkipConfirmation
)

Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  CHAOS TEST: Redis Outage During Active Trading" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "This test will:" -ForegroundColor Yellow
Write-Host "  1. Kill Redis container" -ForegroundColor Yellow
Write-Host "  2. Wait ${OutageDurationSeconds}s (events buffer to disk)" -ForegroundColor Yellow
Write-Host "  3. Restart Redis" -ForegroundColor Yellow
Write-Host "  4. Verify all 5 fixes work correctly" -ForegroundColor Yellow
Write-Host ""

if (-not $SkipConfirmation) {
    $confirmation = Read-Host "Continue? (y/N)"
    if ($confirmation -ne 'y') {
        Write-Host "Test cancelled" -ForegroundColor Red
        exit 1
    }
}

# Setup
$logFile = "data/chaos_test_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
$bufferFile = "data/eventbus_buffer.jsonl"

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  PHASE 1: Pre-Test Verification" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan

# Check if backend is running
Write-Host "[1/3] Checking if backend is running..." -ForegroundColor White
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 5
    if ($response.StatusCode -eq 200) {
        Write-Host "  ✓ Backend is running" -ForegroundColor Green
    }
} catch {
    Write-Host "  ✗ Backend not running or not healthy" -ForegroundColor Red
    Write-Host "  Run: docker compose up -d" -ForegroundColor Yellow
    exit 1
}

# Check if Redis is running
Write-Host "[2/3] Checking if Redis is running..." -ForegroundColor White
$redisContainer = docker ps --filter "name=redis" --format "{{.Names}}" | Select-Object -First 1
if ($redisContainer) {
    Write-Host "  ✓ Redis container: $redisContainer" -ForegroundColor Green
} else {
    Write-Host "  ✗ Redis container not found" -ForegroundColor Red
    exit 1
}

# Delete old buffer file if exists
Write-Host "[3/3] Cleaning up old buffer file..." -ForegroundColor White
if (Test-Path $bufferFile) {
    Remove-Item $bufferFile -Force
    Write-Host "  ✓ Deleted old buffer: $bufferFile" -ForegroundColor Green
} else {
    Write-Host "  ✓ No old buffer file" -ForegroundColor Green
}

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  PHASE 2: Redis Outage Simulation" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan

# Record start time
$startTime = Get-Date

Write-Host ""
Write-Host "🔥 KILLING REDIS CONTAINER..." -ForegroundColor Red
docker stop $redisContainer | Out-Null
Write-Host "  ✓ Redis stopped at $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Yellow

# Monitor logs for trading gate
Write-Host ""
Write-Host "📊 Monitoring logs for trading gate (checking every 5s)..." -ForegroundColor White
Start-Sleep -Seconds 5

# Check backend logs for trading gate
$logs = docker compose logs --tail=50 backend | Select-String "TRADING GATE|Redis.*unavailable|health.*fail"
if ($logs) {
    Write-Host "  ✓ Trading gate detected in logs:" -ForegroundColor Green
    $logs | ForEach-Object { Write-Host "    $_" -ForegroundColor Gray }
} else {
    Write-Host "  ⚠ No trading gate messages found (check manually)" -ForegroundColor Yellow
}

# Check for buffer file
Write-Host ""
Write-Host "📁 Checking for event buffer file..." -ForegroundColor White
Start-Sleep -Seconds 10
if (Test-Path $bufferFile) {
    $lineCount = (Get-Content $bufferFile).Count
    Write-Host "  ✓ Events buffering to disk: $lineCount events" -ForegroundColor Green
} else {
    Write-Host "  ⚠ No buffer file yet (events might not be generated)" -ForegroundColor Yellow
}

# Wait for outage duration
$remainingSeconds = $OutageDurationSeconds - 15  # Already waited 15s
Write-Host ""
Write-Host "⏳ Waiting ${remainingSeconds}s for outage simulation..." -ForegroundColor White
for ($i = $remainingSeconds; $i -gt 0; $i -= 10) {
    Write-Host "  T-${i}s: Redis still down..." -ForegroundColor Gray
    Start-Sleep -Seconds ([Math]::Min(10, $i))
}

Write-Host ""
Write-Host "🔄 RESTARTING REDIS CONTAINER..." -ForegroundColor Green
docker start $redisContainer | Out-Null
Write-Host "  ✓ Redis restarted at $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Yellow

# Record recovery time
$recoveryTime = Get-Date

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  PHASE 3: Post-Recovery Validation" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan

Write-Host ""
Write-Host "⏳ Waiting 10s for system to recover..." -ForegroundColor White
Start-Sleep -Seconds 10

# Check for replay in logs
Write-Host ""
Write-Host "[FIX #4] Checking event replay ordering..." -ForegroundColor White
$replayLogs = docker compose logs --tail=100 backend | Select-String "replay|chronological|buffered_at"
if ($replayLogs) {
    Write-Host "  ✓ Event replay detected:" -ForegroundColor Green
    $replayLogs | Select-Object -First 5 | ForEach-Object { Write-Host "    $_" -ForegroundColor Gray }
} else {
    Write-Host "  ⚠ No replay logs found (check manually)" -ForegroundColor Yellow
}

# Check for cache invalidation
Write-Host ""
Write-Host "[FIX #5] Checking cache invalidation..." -ForegroundColor White
$cacheLogs = docker compose logs --tail=50 backend | Select-String "invalidating.*cache|redis_recovered"
if ($cacheLogs) {
    Write-Host "  ✓ Cache invalidation detected:" -ForegroundColor Green
    $cacheLogs | ForEach-Object { Write-Host "    $_" -ForegroundColor Gray }
} else {
    Write-Host "  ⚠ No cache invalidation logs (check manually)" -ForegroundColor Yellow
}

# Check for position reconciliation
Write-Host ""
Write-Host "[FIX #3] Checking position reconciliation..." -ForegroundColor White
$reconLogs = docker compose logs --tail=50 backend | Select-String "reconcil|positions.*synced"
if ($reconLogs) {
    Write-Host "  ✓ Position reconciliation detected:" -ForegroundColor Green
    $reconLogs | ForEach-Object { Write-Host "    $_" -ForegroundColor Gray }
} else {
    Write-Host "  ⚠ No reconciliation logs (might not have positions)" -ForegroundColor Yellow
}

# Check if buffer file was cleared
Write-Host ""
Write-Host "📁 Checking if buffer file was cleared..." -ForegroundColor White
if (-not (Test-Path $bufferFile)) {
    Write-Host "  ✓ Buffer file cleared after replay" -ForegroundColor Green
} else {
    Write-Host "  ⚠ Buffer file still exists (replay might have failed)" -ForegroundColor Yellow
}

# Check backend health
Write-Host ""
Write-Host "🏥 Checking backend health..." -ForegroundColor White
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 5
    if ($response.StatusCode -eq 200) {
        Write-Host "  ✓ Backend healthy" -ForegroundColor Green
    }
} catch {
    Write-Host "  ✗ Backend not healthy" -ForegroundColor Red
}

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  TEST SUMMARY" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan

$totalDuration = ($recoveryTime - $startTime).TotalSeconds
Write-Host ""
Write-Host "⏱ Outage Duration: ${totalDuration}s" -ForegroundColor White
Write-Host ""

Write-Host "Validation Checklist:" -ForegroundColor White
Write-Host "  [?] FIX #1: Trading stopped during outage (check logs for 'TRADING GATE')" -ForegroundColor Yellow
Write-Host "  [?] FIX #2: Orders retried with exponential backoff (manual test needed)" -ForegroundColor Yellow
Write-Host "  [?] FIX #3: Positions reconciled with Binance (check logs above)" -ForegroundColor Yellow
Write-Host "  [?] FIX #4: Events replayed in chronological order (check logs above)" -ForegroundColor Yellow
Write-Host "  [?] FIX #5: Cache invalidated on recovery (check logs above)" -ForegroundColor Yellow

Write-Host ""
Write-Host "Full logs saved to: $logFile" -ForegroundColor Gray
docker compose logs backend > $logFile

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "Manual Verification Steps:" -ForegroundColor Cyan
Write-Host "1. Check logs: docker compose logs backend | grep -E 'TRADING GATE|replay|cache|reconcile'" -ForegroundColor Gray
Write-Host "2. Verify policy: curl http://localhost:8000/api/policy | jq '.active_mode'" -ForegroundColor Gray
Write-Host "3. Check event buffer: cat data/eventbus_buffer.jsonl | jq '.buffered_at' | sort" -ForegroundColor Gray
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
