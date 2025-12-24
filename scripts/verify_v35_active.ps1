# Verify ExitBrain v3.5 Adaptive Leverage is Active
# Created: 2025-12-24

$VPS = "root@46.224.116.254"
$KEY = "~/.ssh/hetzner_fresh"

Write-Host "🔍 VERIFYING EXITBRAIN v3.5 ADAPTIVE LEVERAGE ACTIVATION" -ForegroundColor Cyan
Write-Host "=" * 70

# Test 1: Check for v3.5 log entries
Write-Host "`n📊 Test 1: Searching backend logs for ExitBrain v3.5 activity..." -ForegroundColor Yellow
$logs = ssh -i $KEY $VPS "docker logs --tail 500 quantum_backend 2>&1"

$v35_logs = $logs | Select-String -Pattern "ExitBrain v3.5|compute_adaptive_levels|v35_integration|adaptive_levels" -CaseSensitive:$false

if ($v35_logs) {
    Write-Host "✅ FOUND v3.5 activity:" -ForegroundColor Green
    $v35_logs | Select-Object -First 10 | ForEach-Object { Write-Host "  $_" }
} else {
    Write-Host "❌ NO v3.5 activity found in logs" -ForegroundColor Red
}

# Test 2: Check for adaptive leverage logs
Write-Host "`n📊 Test 2: Searching for adaptive leverage calculations..." -ForegroundColor Yellow
$leverage_logs = $logs | Select-String -Pattern "leverage|tp1|tp2|tp3|harvest_scheme" -CaseSensitive:$false

if ($leverage_logs) {
    Write-Host "✅ FOUND leverage logs:" -ForegroundColor Green
    $leverage_logs | Select-Object -First 10 | ForEach-Object { Write-Host "  $_" }
} else {
    Write-Host "❌ NO leverage logs found" -ForegroundColor Red
}

# Test 3: Check for ILF metadata logs
Write-Host "`n📊 Test 3: Searching for ILF metadata..." -ForegroundColor Yellow
$ilf_logs = $logs | Select-String -Pattern "ILF|ilf_metadata|volatility_factor|atr_value" -CaseSensitive:$false

if ($ilf_logs) {
    Write-Host "✅ FOUND ILF metadata:" -ForegroundColor Green
    $ilf_logs | Select-Object -First 10 | ForEach-Object { Write-Host "  $_" }
} else {
    Write-Host "❌ NO ILF metadata found" -ForegroundColor Red
}

# Test 4: Check if exitbrain.adaptive_levels stream exists
Write-Host "`n📊 Test 4: Checking exitbrain.adaptive_levels stream..." -ForegroundColor Yellow
$stream_length = ssh -i $KEY $VPS "docker exec quantum_redis redis-cli XLEN quantum:stream:exitbrain.adaptive_levels 2>&1"

if ($stream_length -match "^\d+$") {
    Write-Host "✅ Stream exists with $stream_length events" -ForegroundColor Green
} else {
    Write-Host "⚠️  Stream does not exist or error: $stream_length" -ForegroundColor Yellow
}

# Test 5: Check recent trade.intent events for leverage values
Write-Host "`n📊 Test 5: Checking recent trade.intent events for leverage..." -ForegroundColor Yellow
$trade_intents = ssh -i $KEY $VPS "docker exec quantum_redis redis-cli XREVRANGE quantum:stream:trade.intent + - COUNT 5 2>&1"

if ($trade_intents) {
    Write-Host "✅ Recent trade.intent events:" -ForegroundColor Green
    
    # Parse for leverage values
    $leverage_values = $trade_intents | Select-String -Pattern '"leverage":\s*(\d+\.?\d*)' -AllMatches
    
    if ($leverage_values) {
        Write-Host "  Found leverage values:" -ForegroundColor Cyan
        $leverage_values | ForEach-Object {
            $_.Matches | ForEach-Object {
                $lev = $_.Groups[1].Value
                if ([double]$lev -gt 1) {
                    Write-Host "    ✅ target_leverage: $lev (adaptive!)" -ForegroundColor Green
                } else {
                    Write-Host "    ❌ target_leverage: $lev (stuck at 1?)" -ForegroundColor Red
                }
            }
        }
    } else {
        Write-Host "  ⚠️  No leverage values found in events" -ForegroundColor Yellow
        Write-Host "  Raw data:" -ForegroundColor Gray
        $trade_intents | Select-Object -First 20 | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
    }
} else {
    Write-Host "❌ Could not fetch trade.intent events" -ForegroundColor Red
}

# Test 6: Check for trade_intent subscriber startup
Write-Host "`n📊 Test 6: Checking if TradeIntentSubscriber started..." -ForegroundColor Yellow
$subscriber_logs = $logs | Select-String -Pattern "TradeIntentSubscriber|trade_intent_subscriber|Phase 3.5" -CaseSensitive:$false

if ($subscriber_logs) {
    Write-Host "✅ FOUND subscriber startup:" -ForegroundColor Green
    $subscriber_logs | Select-Object -First 5 | ForEach-Object { Write-Host "  $_" }
} else {
    Write-Host "❌ NO subscriber startup logs found" -ForegroundColor Red
}

Write-Host "`n" + "=" * 70
Write-Host "🏁 VERIFICATION COMPLETE" -ForegroundColor Cyan
Write-Host ""
Write-Host "📋 Summary:" -ForegroundColor White
Write-Host "  - v3.5 activity logs: $(if ($v35_logs) { 'YES ✅' } else { 'NO ❌' })"
Write-Host "  - Leverage calculations: $(if ($leverage_logs) { 'YES ✅' } else { 'NO ❌' })"
Write-Host "  - ILF metadata: $(if ($ilf_logs) { 'YES ✅' } else { 'NO ❌' })"
Write-Host "  - Adaptive stream: $(if ($stream_length -match '^\d+$') { "YES ($stream_length events) ✅" } else { 'NO ❌' })"
Write-Host "  - Subscriber started: $(if ($subscriber_logs) { 'YES ✅' } else { 'NO ❌' })"
Write-Host ""

# Final verdict
$all_tests_passed = $v35_logs -and $leverage_logs -and $ilf_logs -and $subscriber_logs

if ($all_tests_passed) {
    Write-Host "🎉 VERDICT: ExitBrain v3.5 is ACTIVE and processing!" -ForegroundColor Green
} else {
    Write-Host "⚠️  VERDICT: ExitBrain v3.5 may NOT be active" -ForegroundColor Red
    Write-Host "   Check logs above for details" -ForegroundColor Yellow
}
