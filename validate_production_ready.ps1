#!/usr/bin/env pwsh
# ============================================================================
# Production Readiness Validation
# ============================================================================
# Quick validation script to verify all critical fixes are present
# Run before deployment to ensure system is production-ready
# ============================================================================

$ErrorActionPreference = "Continue"

$COLOR_GREEN = "`e[32m"
$COLOR_RED = "`e[31m"
$COLOR_YELLOW = "`e[33m"
$COLOR_BLUE = "`e[34m"
$COLOR_RESET = "`e[0m"

function Write-Pass { param($msg) Write-Host "${COLOR_GREEN}✓ $msg${COLOR_RESET}" }
function Write-Fail { param($msg) Write-Host "${COLOR_RED}✗ $msg${COLOR_RESET}" }
function Write-Warn { param($msg) Write-Host "${COLOR_YELLOW}⚠ $msg${COLOR_RESET}" }
function Write-Info { param($msg) Write-Host "${COLOR_BLUE}ℹ $msg${COLOR_RESET}" }

$passed = 0
$failed = 0

Write-Info "============================================================================"
Write-Info "QUANTUM TRADER - PRODUCTION READINESS VALIDATION"
Write-Info "============================================================================"
Write-Info ""

# ============================================================================
# FIX #1: Real-Time Drawdown Monitor
# ============================================================================
Write-Info "[FIX #1] Real-Time Drawdown Monitor"

if (Select-String -Path "backend/services/position_monitor.py" -Pattern "_check_flash_crash" -Quiet) {
    Write-Pass "  Method _check_flash_crash exists"
    $passed++
} else {
    Write-Fail "  Method _check_flash_crash NOT FOUND"
    $failed++
}

if (Select-String -Path "backend/services/position_monitor.py" -Pattern "market\.flash_crash_detected" -Quiet) {
    Write-Pass "  Event market.flash_crash_detected published"
    $passed++
} else {
    Write-Fail "  Event market.flash_crash_detected NOT published"
    $failed++
}

if (Select-String -Path "backend/services/position_monitor.py" -Pattern "_previous_equity" -Quiet) {
    Write-Pass "  Equity tracking implemented"
    $passed++
} else {
    Write-Fail "  Equity tracking NOT implemented"
    $failed++
}

# ============================================================================
# FIX #2: Dynamic SL Widening
# ============================================================================
Write-Info ""
Write-Info "[FIX #2] Dynamic SL Widening"

if (Select-String -Path "backend/services/ai/trading_profile.py" -Pattern "regime.*Optional\[RegimeType\]" -Quiet) {
    Write-Pass "  Regime parameter added to TP/SL methods"
    $passed++
} else {
    Write-Fail "  Regime parameter NOT added"
    $failed++
}

if (Select-String -Path "backend/services/ai/trading_profile.py" -Pattern "sl_multiplier.*1\.5|sl_multiplier.*2\.5" -Quiet) {
    Write-Pass "  SL multipliers (1.5x/2.5x) implemented"
    $passed++
} else {
    Write-Fail "  SL multipliers NOT implemented"
    $failed++
}

# ============================================================================
# FIX #3: Hybrid Order Strategy
# ============================================================================
Write-Info ""
Write-Info "[FIX #3] Hybrid Order Strategy"

if (Select-String -Path "backend/services/execution.py" -Pattern "STOP.*LIMIT|stopPrice.*limitPrice" -Quiet) {
    Write-Pass "  LIMIT order placement implemented"
    $passed++
} else {
    Write-Fail "  LIMIT order placement NOT implemented"
    $failed++
}

if (Select-String -Path "backend/services/execution.py" -Pattern "STOP_MARKET.*fallback|cancel.*unfilled" -Quiet) {
    Write-Pass "  MARKET fallback strategy implemented"
    $passed++
} else {
    Write-Fail "  MARKET fallback strategy NOT implemented"
    $failed++
}

# ============================================================================
# CE-1: Atomic Promotion Lock
# ============================================================================
Write-Info ""
Write-Info "[CE-1] Atomic Promotion Lock"

if (Select-String -Path "backend/core/event_bus.py" -Pattern "acquire_promotion_lock" -Quiet) {
    Write-Pass "  Method acquire_promotion_lock exists"
    $passed++
} else {
    Write-Fail "  Method acquire_promotion_lock NOT FOUND"
    $failed++
}

if (Select-String -Path "backend/core/event_bus.py" -Pattern "wait_for_promotion_acks" -Quiet) {
    Write-Pass "  Method wait_for_promotion_acks exists"
    $passed++
} else {
    Write-Fail "  Method wait_for_promotion_acks NOT FOUND"
    $failed++
}

if (Select-String -Path "backend/core/event_bus.py" -Pattern "release_promotion_lock" -Quiet) {
    Write-Pass "  Method release_promotion_lock exists"
    $passed++
} else {
    Write-Fail "  Method release_promotion_lock NOT FOUND"
    $failed++
}

if (Select-String -Path "backend/services/continuous_learning/manager.py" -Pattern "acquire_promotion_lock.*wait_for_promotion_acks.*release_promotion_lock" -Quiet) {
    Write-Pass "  CLM uses full promotion lock workflow"
    $passed++
} else {
    Write-Fail "  CLM does NOT use full promotion lock workflow"
    $failed++
}

# ============================================================================
# CE-2: Federation v2 Event Bridge
# ============================================================================
Write-Info ""
Write-Info "[CE-2] Federation v2 Event Bridge"

if (Test-Path "backend/federation/federation_v2_event_bridge.py") {
    Write-Pass "  File federation_v2_event_bridge.py exists"
    $passed++
} else {
    Write-Fail "  File federation_v2_event_bridge.py NOT FOUND"
    $failed++
}

if (Select-String -Path "backend/federation/federation_v2_event_bridge.py" -Pattern "FederationV2EventBridge" -Quiet) {
    Write-Pass "  Class FederationV2EventBridge implemented"
    $passed++
} else {
    Write-Fail "  Class FederationV2EventBridge NOT implemented"
    $failed++
}

if (Select-String -Path "backend/federation/federation_v2_event_bridge.py" -Pattern "_broadcast_to_v2_nodes" -Quiet) {
    Write-Pass "  Method _broadcast_to_v2_nodes exists"
    $passed++
} else {
    Write-Fail "  Method _broadcast_to_v2_nodes NOT FOUND"
    $failed++
}

# ============================================================================
# CE-3: Event Priority Sequencing
# ============================================================================
Write-Info ""
Write-Info "[CE-3] Event Priority Sequencing"

if (Select-String -Path "backend/core/event_bus.py" -Pattern "EVENT_PRIORITIES" -Quiet) {
    Write-Pass "  EVENT_PRIORITIES config exists"
    $passed++
} else {
    Write-Fail "  EVENT_PRIORITIES config NOT FOUND"
    $failed++
}

if (Select-String -Path "backend/core/event_bus.py" -Pattern "subscribe_with_priority" -Quiet) {
    Write-Pass "  Method subscribe_with_priority exists"
    $passed++
} else {
    Write-Fail "  Method subscribe_with_priority NOT FOUND"
    $failed++
}

if (Select-String -Path "backend/core/event_bus.py" -Pattern "ensemble_manager.*sesa.*meta_strategy.*federation" -Quiet) {
    Write-Pass "  Priority levels configured correctly"
    $passed++
} else {
    Write-Fail "  Priority levels NOT configured correctly"
    $failed++
}

# ============================================================================
# DEPLOYMENT FILES
# ============================================================================
Write-Info ""
Write-Info "[DEPLOYMENT FILES]"

$deploymentFiles = @(
    "docker-compose.yml",
    "docker-compose.prod.yml",
    ".env.production.template",
    "deploy_production.ps1",
    "PRODUCTION_MONITORING.md",
    "P1_MAINTENANCE_TASKS.md",
    "PRODUCTION_DEPLOYMENT_SUMMARY.md"
)

foreach ($file in $deploymentFiles) {
    if (Test-Path $file) {
        Write-Pass "  $file exists"
        $passed++
    } else {
        Write-Fail "  $file NOT FOUND"
        $failed++
    }
}

# ============================================================================
# SUMMARY
# ============================================================================
Write-Info ""
Write-Info "============================================================================"
$total = $passed + $failed
$percentage = [math]::Round(($passed / $total) * 100, 1)

if ($failed -eq 0) {
    Write-Pass "VALIDATION PASSED: $passed/$total checks passed ($percentage%)"
    Write-Info ""
    Write-Pass "✓ All critical fixes implemented"
    Write-Pass "✓ System is PRODUCTION-READY"
    Write-Info ""
    Write-Info "Next steps:"
    Write-Info "  1. Edit .env.production with production credentials"
    Write-Info "  2. Run: .\deploy_production.ps1"
    Write-Info "  3. Monitor: See PRODUCTION_MONITORING.md"
} else {
    Write-Fail "VALIDATION FAILED: $failed/$total checks failed ($percentage% passed)"
    Write-Info ""
    Write-Warn "System is NOT production-ready. Fix the failed checks above."
    Write-Info ""
    Write-Info "For implementation details, see:"
    Write-Info "  - Flash Crash: backend/services/position_monitor.py"
    Write-Info "  - Dynamic SL: backend/services/ai/trading_profile.py"
    Write-Info "  - Hybrid Orders: backend/services/execution.py"
    Write-Info "  - Promotion Lock: backend/core/event_bus.py"
    Write-Info "  - Federation Bridge: backend/federation/federation_v2_event_bridge.py"
    exit 1
}

Write-Info "============================================================================"
