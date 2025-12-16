#!/usr/bin/env python3
"""
🧹 CLEANUP SCRIPT - ARCHIVE OLD FILES
Move unused files to archive folder
"""
import os
import shutil
from pathlib import Path
from datetime import datetime

ROOT = Path(r"c:\quantum_trader")
ARCHIVE_DIR = ROOT / f"_archive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Files to remove (from analyzer)
FILES_TO_ARCHIVE = {
    "temporary_fixes": [
        "auto_set_tpsl.py", "docker_force_leverage.py", "emergency_close_losers.py",
        "emergency_fix.py", "fix_backend_db.py", "fix_dashusdt_tpsl.py",
        "fix_dogeusdt.py", "fix_hypeusdt_tpsl.py", "fix_jctusdt_tpsl.py",
        "fix_leverage_proper.py", "fix_ltcusdt_tpsl.py", "fix_metusdt_tpsl.py",
        "fix_missing_sl.py", "fix_paxgusdt_sl.py", "fix_pumpusdt_sl.py",
        "fix_taousdt_tpsl.py", "fix_xanusdt_emergency.py", "fix_xanusdt_sl.py",
        "force_leverage_10x.py", "set_leverage_10x.py", "set_20x_leverage.py"
    ],
    "diagnostic_scripts": [
        "check_ai_integration.py", "check_balance_exposure.py", "check_binance_balance.py",
        "check_binance_positions_leverage.py", "check_dataset.py", "check_execution_journal.py",
        "check_failed.py", "check_features.py", "check_filled.py",
        "check_historical_data.py", "check_orders.py", "check_outcome.py",
        "check_portfolio.py", "check_position_age.py", "check_position_fields.py",
        "check_positions_now.py", "check_positions_state.py", "check_recent_sample.py",
        "check_samples.py", "check_skips.py", "check_status.py",
        "check_symbols.py", "check_xanusdt.py", "diagnose_issues.py",
        "diagnose_model.py", "show_20x_status.py", "show_aggressive_config.py",
        "show_ai_positions.py", "show_tpsl_config.py", "verify_dashboard_integration.py",
        "verify_db.py", "verify_live_config.py", "demo_integration.py"
    ],
    "close_scripts": [
        "close_all_for_fresh_start.py", "close_all_losers.py", 
        "close_all_positions.py", "close_doge.py", "close_xplus.py"
    ],
    "monitoring_old": [
        "live_ai_monitor.py", "monitor_ai.ps1", "monitor_ai.py",
        "monitor_system.ps1", "monitor_tpsl_live.ps1", "monitor_trailing.py",
        "monitor_trailing_live.ps1", "monitor_winrate.ps1",
        "trading_status_summary.py", "watch_ai_live.py"
    ],
    "training_standalone": [
        "continuous_training.py", "continuous_training_perfect.py",
        "main_train_and_backtest.py", "optimize_win_rate.py",
        "retrain_now.py", "train_ai.py", "train_continuous.py",
        "train_custom.py", "train_ensemble.py", "train_ensemble_real_data.py",
        "train_futures_ai.py", "train_futures_master.py", "train_once.py",
        "train_tft.py", "train_tft_backup.py", "train_tft_fixed.py"
    ],
    "backfill": [
        "backfill_binance_history.py", "backfill_training_data.py",
        "bootstrap_training_data.py", "coingecko_backfill.py",
        "futures_backfill.py", "futures_mega_backfill.py",
        "futures_ultra_backfill.py", "generate_historical_samples.py",
        "mega_backfill.py", "multi_exchange_backfill.py",
        "regenerate_dataset.py", "ultra_backfill.py"
    ],
    "test_files": [
        "add_dummy_features.py", "analyze_dataset.py", "direct_test_signals.py",
        "simple_test_signals.py", "test_agent_integration.py",
        "test_agent_predictions.py", "test_ai_dynamic_tpsl.py",
        "test_ai_integration.py", "test_ai_predictions.py",
        "test_ai_signals.py", "test_all_improvements.py",
        "test_api_bulletproof.py", "test_binance_api.py",
        "test_binance_connection.py", "test_bulletproof_api.py",
        "test_confidence_tiers.py", "test_database_bulletproof.py",
        "test_dynamic_keys.py", "test_end_to_end.py",
        "test_full_pipeline.py", "test_full_system.py",
        "test_futures_api.py", "test_futures_balance.py",
        "test_futures_config.py", "test_futures_order_dryrun.py",
        "test_heuristic.py", "test_integration_dashboard_keys.py",
        "test_live_bulletproof.py", "test_live_signals.py",
        "test_model_predictions.py", "test_multi_market_bot.py",
        "test_normalization.py", "test_position_monitor.py",
        "test_precision.py", "test_predictions.py",
        "test_prediction_live.py", "test_retrain.py",
        "test_tft_model.py", "test_tft_predictions.py",
        "conftest.py"
    ],
    "old_docs": [
        "AGGRESSIVE_MODE_PLAN_NOV19.md", "AGGRESSIVE_TRADING_REPORT_NOV19_2025.md",
        "AI_PREDICTION_FIX.md", "AUTO_START_COMPLETE.md",
        "CONTINUOUS_LEARNING_STATUS.md", "DOCKER_TEST_RESULTS.md",
        "END_TO_END_TEST_RESULTS.md", "ENSEMBLE_FIX_COMPLETE.md",
        "FIXES_COMPLETED_NOV19_2025.md", "MIGRATION_PLAN.md",
        "PHASE1_EXACT_CHANGES.md", "PHASE_IMPLEMENTATION_SCRIPT.md",
        "PULL_REQUEST_DRAFT.md", "PULL_REQUEST_WIP_GHCR_USE_GITHUB_TOKEN.md",
        "SESSION_SUMMARY_2025_11_18.md", "SKLEARN_STARTUP_COMPLETE.md",
        "SYSTEM_HEALTH_REPORT_NOV19_2025.md", "TEAM_NOTIFICATION.md",
        "TP_SL_FIX_NOV19_2025.md", "ULTRA_AGGRESSIVE_MODE_ACTIVE.md",
        "ULTRA_AGGRESSIVE_UPDATE_NOV19_2025.md", "WHERE_DID_TRAINING_GO.md"
    ],
    "scripts_old": [
        "check-containers.ps1", "check_new_order.ps1", "check_training_status.ps1",
        "rebuild-docker.ps1", "restart_backend.bat", "restart_training.bat",
        "start_all.bat", "start_training.ps1", "test-docker.ps1",
        "test_ai.ps1", "test_dashboard_integration.ps1", "train_ai_model.bat"
    ],
    "temp_data": [
        "backend_live.log", "continuous_learning.db", "health_dump.json",
        "query", "run18066333212_jobs.json", "temp_health.json",
        "tft_training.log", "tft_training.txt", "tft_training_040909.log",
        "tft_training_full.log", "tft_training_log.txt",
        "tmp_jobs_18066605412.json", "tmp_ws_connect.py"
    ],
    "misc": [
        "backtest_results.csv", "backtest_with_improvements.py",
        "increase_paper_equity.py", "quick_positions.py",
        "set_outcomes.py", "set_tpsl_now.py", "set_tpsl_protection.py",
        "sitecustomize.py", "trades.db", "workflow_at_sha.yml"
    ]
}

def archive_files():
    """Archive files to backup folder"""
    # Create archive directory
    ARCHIVE_DIR.mkdir(exist_ok=True)
    
    archived = 0
    failed = []
    
    print("=" * 80)
    print("🧹 CLEANING UP QUANTUM TRADER")
    print("=" * 80)
    print(f"\n📦 Archive location: {ARCHIVE_DIR}")
    print()
    
    for category, files in FILES_TO_ARCHIVE.items():
        print(f"\n📁 {category.upper().replace('_', ' ')}...")
        
        # Create category folder
        cat_dir = ARCHIVE_DIR / category
        cat_dir.mkdir(exist_ok=True)
        
        for filename in files:
            source = ROOT / filename
            if source.exists():
                try:
                    dest = cat_dir / filename
                    shutil.move(str(source), str(dest))
                    print(f"   [OK] {filename}")
                    archived += 1
                except Exception as e:
                    print(f"   ❌ {filename}: {e}")
                    failed.append(filename)
    
    print("\n" + "=" * 80)
    print(f"[OK] Cleanup complete!")
    print(f"   • Archived: {archived} files")
    print(f"   • Failed: {len(failed)} files")
    print(f"   • Archive: {ARCHIVE_DIR}")
    print("=" * 80)
    
    if failed:
        print(f"\n[WARNING]  Failed to archive: {', '.join(failed)}")

if __name__ == "__main__":
    print("\n[WARNING]  WARNING: This will move {total} files to archive folder!")
    total = sum(len(files) for files in FILES_TO_ARCHIVE.values())
    response = input(f"\nMove {total} files to archive? (yes/no): ")
    
    if response.lower() == "yes":
        archive_files()
    else:
        print("\n❌ Cleanup cancelled")
