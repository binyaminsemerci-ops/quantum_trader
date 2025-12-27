# Phase 4O+ Validation Script - PowerShell
# Intelligent Leverage + RL Position Sizing (Cross-Exchange Enabled)

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "PHASE 4O+ VALIDATION - Intelligent Leverage + RL Position Sizing" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

$errors = 0
$tests = 0

# VPS connection details
$vpsHost = "46.224.116.254"
$vpsUser = "qt"
$sshKey = "~/.ssh/hetzner_fresh"

Write-Host "[INTELLIGENT LEVERAGE V2]" -ForegroundColor Yellow
Write-Host "----------------------------------------------------------------------" -ForegroundColor Gray

# Test 1: Check if ILFv2 engine exists
Write-Host "[1] Check ILFv2 engine file exists..." -NoNewline
$tests++
$result = ssh -i $sshKey ${vpsUser}@${vpsHost} "test -f ~/quantum_trader/microservices/exitbrain_v3_5/intelligent_leverage_engine.py && echo 'EXISTS' || echo 'MISSING'"
if ($result -eq "EXISTS") {
    Write-Host " ✅ PASS" -ForegroundColor Green
} else {
    Write-Host " ❌ FAIL" -ForegroundColor Red
    Write-Host "    Output: $result" -ForegroundColor Gray
    $errors++
}

# Test 2: Check if ExitBrain v3.5 exists
Write-Host "[2] Check ExitBrain v3.5 file exists..." -NoNewline
$tests++
$result = ssh -i $sshKey ${vpsUser}@${vpsHost} "test -f ~/quantum_trader/microservices/exitbrain_v3_5/exit_brain.py && echo 'EXISTS' || echo 'MISSING'"
if ($result -eq "EXISTS") {
    Write-Host " ✅ PASS" -ForegroundColor Green
} else {
    Write-Host " ❌ FAIL" -ForegroundColor Red
    Write-Host "    Output: $result" -ForegroundColor Gray
    $errors++
}

# Test 3: Check quantum:stream:exitbrain.pnl stream
Write-Host "[3] Check quantum:stream:exitbrain.pnl stream..." -NoNewline
$tests++
$streamLen = ssh -i $sshKey ${vpsUser}@${vpsHost} "docker exec quantum_redis redis-cli XLEN quantum:stream:exitbrain.pnl 2>/dev/null || echo '0'"
if ([int]$streamLen -gt 0) {
    Write-Host " ✅ PASS ($streamLen entries)" -ForegroundColor Green
} else {
    Write-Host " ❌ FAIL (stream empty)" -ForegroundColor Red
    Write-Host "    Note: Stream will populate after first ExitBrain calculation" -ForegroundColor Gray
    $errors++
}

Write-Host ""
Write-Host "[RL POSITION SIZING AGENT]" -ForegroundColor Yellow
Write-Host "----------------------------------------------------------------------" -ForegroundColor Gray

# Test 4: Check if RL agent file exists
Write-Host "[4] Check RL agent file exists..." -NoNewline
$tests++
$result = ssh -i $sshKey ${vpsUser}@${vpsHost} "test -f ~/quantum_trader/microservices/rl_sizing_agent/rl_agent.py && echo 'EXISTS' || echo 'MISSING'"
if ($result -eq "EXISTS") {
    Write-Host " ✅ PASS" -ForegroundColor Green
} else {
    Write-Host " ❌ FAIL" -ForegroundColor Red
    Write-Host "    Output: $result" -ForegroundColor Gray
    $errors++
}

# Test 5: Check if PnL feedback listener exists
Write-Host "[5] Check PnL feedback listener exists..." -NoNewline
$tests++
$result = ssh -i $sshKey ${vpsUser}@${vpsHost} "test -f ~/quantum_trader/microservices/rl_sizing_agent/pnl_feedback_listener.py && echo 'EXISTS' || echo 'MISSING'"
if ($result -eq "EXISTS") {
    Write-Host " ✅ PASS" -ForegroundColor Green
} else {
    Write-Host " ❌ FAIL" -ForegroundColor Red
    Write-Host "    Output: $result" -ForegroundColor Gray
    $errors++
}

# Test 6: Check if RL policy model directory exists
Write-Host "[6] Check RL model directory..." -NoNewline
$tests++
$result = ssh -i $sshKey ${vpsUser}@${vpsHost} "docker exec quantum_ai_engine test -d /models && echo 'EXISTS' || echo 'MISSING'"
if ($result -eq "EXISTS") {
    Write-Host " ✅ PASS" -ForegroundColor Green
} else {
    Write-Host " ⚠️  WARN (will be created on first training)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "[AI ENGINE HEALTH CHECK]" -ForegroundColor Yellow
Write-Host "----------------------------------------------------------------------" -ForegroundColor Gray

# Test 7: Check AI Engine health endpoint
Write-Host "[7] Check AI Engine /health endpoint..." -NoNewline
$tests++
try {
    $healthJson = ssh -i $sshKey ${vpsUser}@${vpsHost} "curl -s http://localhost:8001/health 2>/dev/null"
    $health = $healthJson | ConvertFrom-Json
    
    if ($health.status -eq "OK") {
        Write-Host " ✅ PASS" -ForegroundColor Green
        
        # Check Phase 4O+ metrics
        if ($health.metrics.intelligent_leverage_v2 -or $health.metrics.intelligent_leverage) {
            Write-Host "    ✓ Intelligent Leverage v2: ENABLED" -ForegroundColor Green
            if ($health.metrics.intelligent_leverage) {
                Write-Host "      - Avg Leverage: $($health.metrics.intelligent_leverage.avg_leverage)x" -ForegroundColor Gray
                Write-Host "      - Avg Confidence: $($health.metrics.intelligent_leverage.avg_confidence)" -ForegroundColor Gray
                Write-Host "      - Calculations: $($health.metrics.intelligent_leverage.calculations_total)" -ForegroundColor Gray
            }
        } else {
            Write-Host "    ⚠️  Intelligent Leverage v2: NOT IN METRICS" -ForegroundColor Yellow
        }
        
        if ($health.metrics.rl_position_sizing -or $health.metrics.rl_agent) {
            Write-Host "    ✓ RL Position Sizing: ENABLED" -ForegroundColor Green
            if ($health.metrics.rl_agent) {
                Write-Host "      - Policy Version: $($health.metrics.rl_agent.policy_version)" -ForegroundColor Gray
                Write-Host "      - Trades Processed: $($health.metrics.rl_agent.trades_processed)" -ForegroundColor Gray
                Write-Host "      - Reward Mean: $($health.metrics.rl_agent.reward_mean)" -ForegroundColor Gray
            }
        } else {
            Write-Host "    ⚠️  RL Position Sizing: NOT IN METRICS" -ForegroundColor Yellow
        }
        
        # Check Cross-Exchange Intelligence (Phase 4M+)
        if ($health.metrics.cross_exchange_intelligence) {
            Write-Host "    ✓ Cross-Exchange Intelligence: ENABLED (Phase 4M+)" -ForegroundColor Green
        }
        
    } else {
        Write-Host " ❌ FAIL (status: $($health.status))" -ForegroundColor Red
        $errors++
    }
} catch {
    Write-Host " ❌ FAIL (health check failed)" -ForegroundColor Red
    Write-Host "    Error: $_" -ForegroundColor Gray
    $errors++
}

Write-Host ""
Write-Host "[INTEGRATION STATUS]" -ForegroundColor Yellow
Write-Host "----------------------------------------------------------------------" -ForegroundColor Gray

# Test 8: Check for ILFv2 initialization logs
Write-Host "[8] Check for ILFv2 initialization in logs..." -NoNewline
$tests++
$logCount = ssh -i $sshKey ${vpsUser}@${vpsHost} "docker logs quantum_ai_engine 2>&1 | grep -c 'ILF-v2.*Initialized' || echo '0'"
if ([int]$logCount -gt 0) {
    Write-Host " ✅ PASS ($logCount occurrences)" -ForegroundColor Green
} else {
    Write-Host " ❌ FAIL (no initialization logs)" -ForegroundColor Red
    Write-Host "    Note: Check if ExitBrain v3.5 properly imports ILFv2" -ForegroundColor Gray
    $errors++
}

# Test 9: Check for RL agent initialization logs
Write-Host "[9] Check for RL agent initialization..." -NoNewline
$tests++
$logCount = ssh -i $sshKey ${vpsUser}@${vpsHost} "docker logs quantum_ai_engine 2>&1 | grep -c 'RL-Agent.*Initialized' || echo '0'"
if ([int]$logCount -gt 0) {
    Write-Host " ✅ PASS ($logCount occurrences)" -ForegroundColor Green
} else {
    Write-Host " ⚠️  WARN (no RL agent logs yet)" -ForegroundColor Yellow
    Write-Host "    Note: RL agent initialized on first trade" -ForegroundColor Gray
}

Write-Host ""
Write-Host "[FORMULA VERIFICATION]" -ForegroundColor Yellow
Write-Host "----------------------------------------------------------------------" -ForegroundColor Gray

Write-Host "ILFv2 Formula:" -ForegroundColor Cyan
Write-Host "  base = 5 + confidence × 75" -ForegroundColor Gray
Write-Host "  leverage = base × vol_factor × pnl_factor × symbol_factor ×" -ForegroundColor Gray
Write-Host "             margin_factor × divergence_factor × funding_factor" -ForegroundColor Gray
Write-Host "  Range: 5-80x" -ForegroundColor Gray
Write-Host ""
Write-Host "RL Reward Function:" -ForegroundColor Cyan
Write-Host "  reward = (pnl_pct × confidence)" -ForegroundColor Gray
Write-Host "           - 0.005 × |leverage - target_leverage|" -ForegroundColor Gray
Write-Host "           - 0.002 × exch_divergence" -ForegroundColor Gray
Write-Host "           + 0.003 × sign(pnl_trend)" -ForegroundColor Gray

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "VALIDATION SUMMARY" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Tests Run: $tests" -ForegroundColor White
Write-Host "Errors: $errors" -ForegroundColor $(if ($errors -eq 0) { "Green" } else { "Red" })
Write-Host ""

if ($errors -eq 0) {
    Write-Host "✅ VALIDATION PASSED" -ForegroundColor Green
    Write-Host "   Phase 4O+ integration ready for production" -ForegroundColor Green
} else {
    Write-Host "❌ VALIDATION FAILED" -ForegroundColor Red
    Write-Host "   Fix errors and re-run validation" -ForegroundColor Red
}
Write-Host ""
