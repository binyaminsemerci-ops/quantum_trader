# PATCH-P2-01: Backend Services Refactoring
# ==========================================
# 
# This script migrates backend/services/ from a flat structure to organized subdirectories
# to reduce "god-files" and improve maintainability.

Write-Host "[P2-01] Backend Services Structure Refactoring" -ForegroundColor Cyan
Write-Host "=" * 60

$ErrorActionPreference = "Stop"
$services_root = "backend/services"

# Define file categorization
$migration_map = @{
    "ai" = @(
        "ai_hedgefund_os.py",
        "ai_hfos_integration.py",
        "ai_trading_engine.py",
        "model_supervisor.py",
        "continuous_learning_manager.py",
        "continuous_learning_manager_example.py",
        "rl_action_space_v2.py",
        "rl_episode_tracker_v2.py",
        "rl_position_sizing_agent.py",
        "rl_reward_engine_v2.py",
        "rl_state_manager_v2.py",
        "rl_v3_live_orchestrator.py",
        "rl_v3_training_daemon.py",
        "trading_mathematician.py",
        "math_ai_integration.py",
        "msc_ai_integration.py",
        "msc_ai_scheduler.py"
    )
    "risk" = @(
        "risk_guard.py",
        "safety_governor.py",
        "advanced_risk.py",
        "emergency_stop_system.py",
        "ess_alerters.py",
        "ess_integration_example.py",
        "ess_integration_main.py",
        "signal_quality_filter.py",
        "funding_rate_filter.py",
        "safety_policy.py"
    )
    "execution" = @(
        "smart_execution.py",
        "event_driven_executor.py",
        "execution.py",
        "positions.py",
        "position_sizing.py",
        "smart_position_sizer.py",
        "dynamic_tpsl.py",
        "hybrid_tpsl.py",
        "trailing_stop_manager.py",
        "exit_policy_regime_config.py",
        "selection_engine.py",
        "liquidity.py"
    )
    "governance" = @(
        "trade_state_store.py",
        "orchestrator_policy.py",
        "orchestrator_config.py",
        "policy_observer.py",
        "legacy_policy_store.py",
        "policy_store_examples.py",
        "policy_store_integration_demo.py"
    )
    "monitoring" = @(
        "system_health_monitor.py",
        "health_monitor.py",
        "position_monitor.py",
        "symbol_performance.py",
        "logging_extensions.py",
        "integrate_system_health_monitor.py",
        "self_healing.py",
        "recovery_actions.py"
    )
}

# Dry run flag
$dry_run = $false

Write-Host "`n[PHASE 1] Analyzing current structure..." -ForegroundColor Yellow

$total_files = 0
foreach ($category in $migration_map.Keys) {
    $count = $migration_map[$category].Count
    $total_files += $count
    Write-Host "  $category/: $count files" -ForegroundColor Gray
}

Write-Host "`nTotal files to migrate: $total_files" -ForegroundColor White

Write-Host "`n[PHASE 2] Migration Preview (DRY RUN)" -ForegroundColor Yellow

foreach ($category in $migration_map.Keys) {
    Write-Host "`n  Category: $category/" -ForegroundColor Cyan
    
    foreach ($file in $migration_map[$category]) {
        $source = Join-Path $services_root $file
        $dest = Join-Path $services_root "$category/$file"
        
        if (Test-Path $source) {
            Write-Host "    ✓ $file" -ForegroundColor Green
            if (-not $dry_run) {
                # Move file
                Move-Item -Path $source -Destination $dest -Force
                Write-Host "      → Moved to $category/" -ForegroundColor Gray
            }
        } else {
            Write-Host "    ✗ $file (not found)" -ForegroundColor Red
        }
    }
}

Write-Host "`n[PHASE 3] Import Path Updates Required" -ForegroundColor Yellow
Write-Host @"

After migration, update imports across the codebase:

OLD: from backend.services.ai_hedgefund_os import AIHedgeFundOS
NEW: from backend.services.ai.ai_hedgefund_os import AIHedgeFundOS

OLD: from backend.services.risk_guard import RiskGuard
NEW: from backend.services.risk.risk_guard import RiskGuard

OLD: from backend.services.smart_execution import SmartExecutor
NEW: from backend.services.execution.smart_execution import SmartExecutor

OLD: from backend.services.trade_state_store import TradeStateStore
NEW: from backend.services.governance.trade_state_store import TradeStateStore

OLD: from backend.services.system_health_monitor import SystemHealthMonitor
NEW: from backend.services.monitoring.system_health_monitor import SystemHealthMonitor

Run find-and-replace across entire codebase:
  grep -r "from backend.services." --include="*.py" | wc -l
"@

Write-Host "`n[PHASE 4] Next Steps" -ForegroundColor Yellow
Write-Host @"

1. Review migration plan (this dry run)
2. Set `$dry_run = `$false and re-run script
3. Update all imports using find-and-replace
4. Run tests to verify nothing broke
5. Commit changes: "PATCH-P2-01: Refactor services structure"

"@

Write-Host "[P2-01] Dry run complete. Set `$dry_run=`$false to execute." -ForegroundColor Green
