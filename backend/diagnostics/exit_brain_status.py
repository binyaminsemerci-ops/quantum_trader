"""
Exit Brain Status Diagnostics
==============================

Health check endpoint for Exit Brain v3 activation status.

Provides real-time visibility into:
- Current exit mode (LEGACY vs EXIT_BRAIN_V3)
- Executor mode (SHADOW vs LIVE)
- Rollout safety flag status
- Executor running state
- Exit order metrics
"""

from typing import Dict, Any, Optional
from datetime import datetime

from backend.config.exit_mode import (
    get_exit_mode,
    get_exit_executor_mode,
    is_exit_brain_live_rollout_enabled,
    is_exit_brain_live_fully_enabled,
    is_exit_brain_mode
)
from backend.services.execution.exit_order_gateway import get_exit_order_metrics


def get_exit_brain_status(app_state: Optional[Any] = None) -> Dict[str, Any]:
    """
    Get comprehensive Exit Brain v3 status for diagnostics.
    
    Args:
        app_state: Optional FastAPI app.state object to check executor status
        
    Returns:
        Dict with status information
    """
    # Config status
    exit_mode = get_exit_mode()
    executor_mode = get_exit_executor_mode()
    rollout_enabled = is_exit_brain_live_rollout_enabled()
    live_fully_enabled = is_exit_brain_live_fully_enabled()
    
    # Executor status
    executor_running = False
    executor_effective_mode = None
    last_decision_time = None
    
    if app_state and hasattr(app_state, 'exit_brain_executor'):
        executor = app_state.exit_brain_executor
        executor_running = True
        executor_effective_mode = getattr(executor, 'effective_mode', None)
        
        # Try to get last decision timestamp from shadow logs
        try:
            shadow_logs = executor.get_shadow_logs()
            if shadow_logs:
                last_decision_time = shadow_logs[-1].get('timestamp')
        except:
            pass
    
    # Metrics
    metrics = get_exit_order_metrics().get_summary()
    
    # Build status
    status = {
        "timestamp": datetime.utcnow().isoformat(),
        "config": {
            "exit_mode": exit_mode,
            "exit_executor_mode": executor_mode,
            "exit_brain_live_rollout": "ENABLED" if rollout_enabled else "DISABLED",
            "live_mode_active": live_fully_enabled
        },
        "executor": {
            "running": executor_running,
            "effective_mode": executor_effective_mode,
            "last_decision_timestamp": last_decision_time
        },
        "metrics": {
            "total_exit_orders": metrics.get("total_orders", 0),
            "blocked_legacy_orders": metrics.get("blocked_legacy_orders", 0),
            "ownership_conflicts": metrics.get("ownership_conflicts", 0),
            "orders_by_module": metrics.get("orders_by_module", {}),
            "orders_by_kind": metrics.get("orders_by_kind", {})
        },
        "operational_state": _determine_operational_state(
            exit_mode, 
            executor_mode, 
            rollout_enabled,
            executor_running
        )
    }
    
    return status


def _determine_operational_state(
    exit_mode: str,
    executor_mode: str,
    rollout_enabled: bool,
    executor_running: bool
) -> str:
    """
    Determine human-readable operational state.
    
    Returns:
        State string: "LEGACY", "EXIT_BRAIN_SHADOW", "EXIT_BRAIN_LIVE", etc.
    """
    if exit_mode == "LEGACY":
        return "LEGACY"
    
    if exit_mode == "EXIT_BRAIN_V3":
        if not executor_running:
            return "EXIT_BRAIN_V3_NOT_RUNNING"
        
        if executor_mode == "LIVE" and rollout_enabled:
            return "EXIT_BRAIN_V3_LIVE"
        else:
            return "EXIT_BRAIN_V3_SHADOW"
    
    return "UNKNOWN"


def print_exit_brain_status(app_state: Optional[Any] = None):
    """
    Print Exit Brain v3 status to console in human-readable format.
    
    Useful for CLI diagnostics.
    
    Args:
        app_state: Optional FastAPI app.state object
    """
    status = get_exit_brain_status(app_state)
    
    print("\n" + "="*60)
    print("EXIT BRAIN V3 STATUS")
    print("="*60)
    
    # Operational state
    state = status["operational_state"]
    state_emoji = {
        "LEGACY": "🔵",
        "EXIT_BRAIN_V3_SHADOW": "🟡",
        "EXIT_BRAIN_V3_LIVE": "🔴",
        "EXIT_BRAIN_V3_NOT_RUNNING": "⚠️"
    }.get(state, "❓")
    
    print(f"\n{state_emoji} Operational State: {state}")
    
    # Config
    print(f"\n📋 Configuration:")
    print(f"  EXIT_MODE: {status['config']['exit_mode']}")
    print(f"  EXIT_EXECUTOR_MODE: {status['config']['exit_executor_mode']}")
    print(f"  EXIT_BRAIN_V3_LIVE_ROLLOUT: {status['config']['exit_brain_live_rollout']}")
    print(f"  Live Mode Active: {status['config']['live_mode_active']}")
    
    # Executor
    print(f"\n🤖 Executor:")
    print(f"  Running: {status['executor']['running']}")
    print(f"  Effective Mode: {status['executor']['effective_mode']}")
    if status['executor']['last_decision_timestamp']:
        print(f"  Last Decision: {status['executor']['last_decision_timestamp']}")
    
    # Metrics
    metrics = status['metrics']
    print(f"\n📊 Metrics:")
    print(f"  Total Exit Orders: {metrics['total_exit_orders']}")
    print(f"  Blocked Legacy Orders: {metrics['blocked_legacy_orders']}")
    print(f"  Ownership Conflicts: {metrics['ownership_conflicts']}")
    
    if metrics['orders_by_module']:
        print(f"\n  Orders by Module:")
        for module, count in sorted(metrics['orders_by_module'].items(), key=lambda x: -x[1]):
            print(f"    {module}: {count}")
    
    if metrics['orders_by_kind']:
        print(f"\n  Orders by Kind:")
        for kind, count in sorted(metrics['orders_by_kind'].items(), key=lambda x: -x[1]):
            print(f"    {kind}: {count}")
    
    print("\n" + "="*60)
    
    # Warnings
    if state == "EXIT_BRAIN_V3_LIVE":
        print("\n⚠️  WARNING: EXIT BRAIN V3 LIVE MODE ACTIVE")
        print("  AI executor is placing real orders.")
        print("  Legacy exit modules are blocked.")
    elif state == "EXIT_BRAIN_V3_NOT_RUNNING":
        print("\n⚠️  WARNING: EXIT_BRAIN_V3 mode but executor not running")
        print("  Check startup logs for errors.")
    
    if metrics['blocked_legacy_orders'] > 0:
        print(f"\n✅ Blocked {metrics['blocked_legacy_orders']} legacy module orders (expected in LIVE mode)")
    
    if metrics['ownership_conflicts'] > 0 and state != "EXIT_BRAIN_V3_LIVE":
        print(f"\n⚠️  Detected {metrics['ownership_conflicts']} ownership conflicts")
        print("  Legacy modules attempting to place exit orders in EXIT_BRAIN_V3 mode.")
    
    print()
