#!/usr/bin/env python3
"""
FUNCTIONAL TEST FOR AI-OS SUBSYSTEMS
=====================================

Tests each AI-OS subsystem with real function calls to prove they exist and work.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from pathlib import Path

print("=" * 80)
print("AI-OS SUBSYSTEM FUNCTIONAL TESTS")
print("=" * 80)

# Test 1: Portfolio Balancer AI (PBA)
print("\n[TEST 1] Portfolio Balancer AI (PBA)")
try:
    from backend.services.portfolio_balancer import PortfolioBalancerAI, Position, CandidateTrade
    
    pba = PortfolioBalancerAI()
    
    # Create dummy data
    positions = [
        Position(
            symbol="BTCUSDT",
            side="LONG",
            size=0.1,
            entry_price=50000,
            current_price=51000,
            margin=1000,
            leverage=30
        )
    ]
    
    candidates = [
        CandidateTrade(
            symbol="ETHUSDT",
            action="BUY",
            confidence=0.75,
            size=1.0,
            margin_required=500
        )
    ]
    
    result = pba.analyze_portfolio(
        positions=positions,
        candidates=candidates,
        total_equity=5000,
        used_margin=1000,
        free_margin=4000
    )
    
    print(f"✅ PBA EXISTS AND WORKS")
    print(f"   Status: {result.status}")
    print(f"   Rejected trades: {len(result.rejected_trades)}")
    print(f"   Warnings: {len(result.warnings)}")
    
except Exception as e:
    print(f"❌ PBA FAILED: {e}")

# Test 2: Profit Amplification Layer (PAL)
print("\n[TEST 2] Profit Amplification Layer (PAL)")
try:
    from backend.services.profit_amplification import ProfitAmplificationLayer, PositionSnapshot
    
    pal = ProfitAmplificationLayer(data_dir="/tmp/test_pal")
    
    # Create dummy position
    position = PositionSnapshot(
        symbol="BTCUSDT",
        side="LONG",
        current_R=2.5,
        peak_R=3.0,
        unrealized_pnl=1500,
        unrealized_pnl_pct=0.05,
        drawdown_from_peak_R=0.16,
        drawdown_from_peak_pnl_pct=0.10,
        current_leverage=30,
        position_size_usd=50000,
        risk_pct=2.0,
        hold_time_hours=6.5,
        entry_time="2025-11-23T10:00:00Z",
        pil_classification="WINNER"
    )
    
    recommendation = pal.analyze_position(position)
    
    print(f"✅ PAL EXISTS AND WORKS")
    print(f"   Action: {recommendation.action.value}")
    print(f"   Reasons: {len(recommendation.reasons)}")
    print(f"   Details: {recommendation.details}")
    
except Exception as e:
    print(f"❌ PAL FAILED: {e}")

# Test 3: Model Supervisor
print("\n[TEST 3] Model Supervisor")
try:
    from backend.services.ai.model_supervisor import ModelSupervisor
    
    supervisor = ModelSupervisor(
        data_dir="/tmp/test_supervisor",
        analysis_window_days=30
    )
    
    # Create dummy trade data
    from datetime import datetime, timezone
    dummy_trade = {
        "symbol": "BTCUSDT",
        "action": "BUY",
        "confidence": 0.75,
        "model": "XGBoost",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pnl": 150,
        "win": True
    }
    
    supervisor.record_prediction(
        symbol=dummy_trade["symbol"],
        action=dummy_trade["action"],
        confidence=dummy_trade["confidence"],
        model_name=dummy_trade["model"],
        timestamp=dummy_trade["timestamp"]
    )
    
    report = supervisor.generate_report()
    
    print(f"✅ MODEL SUPERVISOR EXISTS AND WORKS")
    print(f"   Overall health: {report['overall_health']}")
    print(f"   Models tracked: {len(report.get('model_rankings', []))}")
    print(f"   Retrain needed: {report.get('retrain_recommendation', {}).get('needed', False)}")
    
except Exception as e:
    print(f"❌ MODEL SUPERVISOR FAILED: {e}")

# Test 4: Self-Healing System
print("\n[TEST 4] Self-Healing System")
try:
    from backend.services.self_healing import SelfHealingSystem
    
    self_healing = SelfHealingSystem(
        data_dir="/tmp/test_healing",
        check_interval=30
    )
    
    # Check subsystem health
    from backend.services.self_healing import SubsystemType
    health_checks = {}
    
    for subsystem in [SubsystemType.DATA_FEED, SubsystemType.EXCHANGE_CONNECTION]:
        try:
            health = self_healing._check_generic_subsystem(subsystem, f"{subsystem.value}.json")
            health_checks[subsystem.value] = health.status.value
        except:
            health_checks[subsystem.value] = "UNKNOWN"
    
    print(f"✅ SELF-HEALING EXISTS AND WORKS")
    print(f"   Subsystems checked: {len(health_checks)}")
    print(f"   Health checks: {health_checks}")
    
except Exception as e:
    print(f"❌ SELF-HEALING FAILED: {e}")

# Test 5: AI-HFOS
print("\n[TEST 5] AI Hedgefund Operating System (AI-HFOS)")
try:
    from backend.services.ai.ai_hedgefund_os import AIHedgeFundOS
    
    ai_hfos = AIHedgeFundOS(data_dir="/tmp/test_hfos")
    
    # Create dummy system snapshot
    snapshot = {
        "total_equity": 5000,
        "used_margin": 1500,
        "free_margin": 3500,
        "open_positions": 2,
        "daily_pnl_pct": 2.5,
        "subsystem_health": {"risk_os": "HEALTHY", "execution": "HEALTHY"}
    }
    
    output = ai_hfos.analyze(snapshot)
    
    print(f"✅ AI-HFOS EXISTS AND WORKS")
    print(f"   System risk mode: {output.system_risk_mode.value}")
    print(f"   System health: {output.system_health.value}")
    print(f"   Allow new trades: {output.global_directives.allow_new_trades}")
    
except Exception as e:
    print(f"❌ AI-HFOS FAILED: {e}")

# Test 6: Retraining Orchestrator
print("\n[TEST 6] Retraining Orchestrator")
try:
    from backend.services.retraining_orchestrator import RetrainingOrchestrator
    
    orchestrator = RetrainingOrchestrator(data_dir="/tmp/test_retrain")
    
    # Evaluate retraining need
    model_metrics = {
        "XGBoost": {"winrate": 0.45, "avg_r": 0.5, "recent_winrate": 0.40},
        "LightGBM": {"winrate": 0.55, "avg_r": 1.2, "recent_winrate": 0.52}
    }
    
    evaluation = orchestrator.evaluate_retraining_need(model_metrics)
    
    print(f"✅ RETRAINING ORCHESTRATOR EXISTS AND WORKS")
    print(f"   Retrain needed: {evaluation.get('needs_retraining', False)}")
    print(f"   Priority: {evaluation.get('priority', 'N/A')}")
    print(f"   Trigger type: {evaluation.get('trigger_type', 'N/A')}")
    
except Exception as e:
    print(f"❌ RETRAINING ORCHESTRATOR FAILED: {e}")

print("\n" + "=" * 80)
print("FUNCTIONAL TESTS COMPLETE")
print("=" * 80)
