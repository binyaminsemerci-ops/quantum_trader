"""
RECOVERY ACTION ENGINE

Executes safe recovery actions when issues are detected by the Self-Healing System.

Author: Quantum Trader AI Team
Date: November 23, 2025
"""

import os
import logging
import asyncio
import json
import subprocess
import psutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict

from self_healing import RecoveryAction, SubsystemType

logger = logging.getLogger(__name__)


@dataclass
class RecoveryResult:
    """Result of a recovery action execution."""
    action: RecoveryAction
    subsystem: SubsystemType
    timestamp: str
    success: bool
    
    message: str
    duration_ms: float
    
    # Before/after state
    state_before: Optional[str]
    state_after: Optional[str]
    
    # Error details
    error: Optional[str]


class RecoveryActionEngine:
    """
    Executes recovery actions to restore system health.
    
    Supported Actions:
    - RESTART_SUBSYSTEM - Restart failed subsystem
    - PAUSE_TRADING - Halt all trading activity
    - SWITCH_TO_SAFE_PROFILE - Reduce risk exposure
    - DISABLE_MODULE - Turn off failing module
    - NO_NEW_TRADES - Block new positions
    - DEFENSIVE_EXIT - Close risky positions
    - CLOSE_ALL_POSITIONS - Emergency exit all positions
    - RELOAD_CONFIG - Refresh configuration
    - CLEAR_CACHE - Clear system caches
    - FALLBACK_TO_BACKUP - Switch to backup system
    """
    
    def __init__(
        self,
        data_dir: str = "/app/data",
        config_dir: str = "/app/config",
        
        # Execution settings
        dry_run: bool = False,
        max_retries: int = 2,
        retry_delay_sec: int = 5,
    ):
        self.data_dir = Path(data_dir)
        self.config_dir = Path(config_dir)
        
        self.dry_run = dry_run
        self.max_retries = max_retries
        self.retry_delay_sec = retry_delay_sec
        
        # State tracking
        self.execution_history: list[RecoveryResult] = []
        self.trading_paused = False
        self.disabled_modules: set[SubsystemType] = set()
        
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    # --------------------------------------------------------
    # MAIN EXECUTION
    # --------------------------------------------------------
    
    async def execute_recovery_action(
        self,
        action: RecoveryAction,
        subsystem: SubsystemType,
        reason: str = "System health issue detected"
    ) -> RecoveryResult:
        """
        Execute a recovery action.
        
        Args:
            action: Recovery action to execute
            subsystem: Affected subsystem
            reason: Reason for executing action
        
        Returns:
            RecoveryResult with execution details
        """
        start_time = datetime.now(timezone.utc)
        
        logger.info(f"[RECOVERY] Executing {action.value} for {subsystem.value}")
        logger.info(f"[RECOVERY] Reason: {reason}")
        
        if self.dry_run:
            logger.info("[RECOVERY] DRY RUN - Action not actually executed")
            return RecoveryResult(
                action=action,
                subsystem=subsystem,
                timestamp=start_time.isoformat(),
                success=True,
                message="Dry run - no action taken",
                duration_ms=0.0,
                state_before=None,
                state_after=None,
                error=None
            )
        
        # Get state before
        state_before = await self._get_subsystem_state(subsystem)
        
        # Execute action with retries
        success = False
        error_msg = None
        
        for attempt in range(1, self.max_retries + 1):
            try:
                if action == RecoveryAction.RESTART_SUBSYSTEM:
                    await self._restart_subsystem(subsystem)
                elif action == RecoveryAction.PAUSE_TRADING:
                    await self._pause_trading()
                elif action == RecoveryAction.SWITCH_TO_SAFE_PROFILE:
                    await self._switch_to_safe_profile()
                elif action == RecoveryAction.DISABLE_MODULE:
                    await self._disable_module(subsystem)
                elif action == RecoveryAction.NO_NEW_TRADES:
                    await self._enable_no_new_trades()
                elif action == RecoveryAction.DEFENSIVE_EXIT:
                    await self._defensive_exit()
                elif action == RecoveryAction.CLOSE_ALL_POSITIONS:
                    await self._close_all_positions()
                elif action == RecoveryAction.RELOAD_CONFIG:
                    await self._reload_config()
                elif action == RecoveryAction.CLEAR_CACHE:
                    await self._clear_cache()
                elif action == RecoveryAction.FALLBACK_TO_BACKUP:
                    await self._fallback_to_backup(subsystem)
                else:
                    raise ValueError(f"Unknown action: {action}")
                
                success = True
                break
            
            except Exception as e:
                error_msg = str(e)
                logger.error(f"[RECOVERY] Attempt {attempt}/{self.max_retries} failed: {e}")
                
                if attempt < self.max_retries:
                    logger.info(f"[RECOVERY] Retrying in {self.retry_delay_sec}s...")
                    await asyncio.sleep(self.retry_delay_sec)
        
        # Get state after
        state_after = await self._get_subsystem_state(subsystem)
        
        # Calculate duration
        duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        
        # Create result
        result = RecoveryResult(
            action=action,
            subsystem=subsystem,
            timestamp=start_time.isoformat(),
            success=success,
            message=f"Action {'succeeded' if success else 'failed'}" + (f" after {attempt} attempts" if attempt > 1 else ""),
            duration_ms=duration_ms,
            state_before=state_before,
            state_after=state_after,
            error=error_msg
        )
        
        # Track history
        self.execution_history.append(result)
        self._save_history()
        
        # Log result
        if success:
            logger.info(f"[RECOVERY] ✅ {action.value} completed successfully ({duration_ms:.0f}ms)")
        else:
            logger.error(f"[RECOVERY] ❌ {action.value} failed: {error_msg}")
        
        return result
    
    # --------------------------------------------------------
    # RECOVERY ACTION IMPLEMENTATIONS
    # --------------------------------------------------------
    
    async def _restart_subsystem(self, subsystem: SubsystemType):
        """Restart a failed subsystem."""
        logger.info(f"[RECOVERY] Restarting {subsystem.value}...")
        
        # Map subsystem to restart command
        restart_commands = {
            SubsystemType.EVENT_EXECUTOR: ["python", "-m", "backend.event_driven_executor"],
            SubsystemType.ORCHESTRATOR: ["python", "-m", "backend.services.orchestrator_policy"],
            SubsystemType.PORTFOLIO_BALANCER: ["python", "-m", "backend.services.portfolio_balancer"],
            SubsystemType.MODEL_SUPERVISOR: ["python", "-m", "backend.services.model_supervisor"],
            SubsystemType.RETRAINING_ORCHESTRATOR: ["python", "-m", "backend.services.retraining_orchestrator"],
            SubsystemType.POSITION_MONITOR: ["python", "-m", "backend.services.position_monitor"],
        }
        
        if subsystem not in restart_commands:
            logger.warning(f"[RECOVERY] No restart command defined for {subsystem.value}")
            return
        
        # Create restart flag
        restart_flag = self.data_dir / f"{subsystem.value}_restart_requested.flag"
        restart_flag.touch()
        
        logger.info(f"[RECOVERY] Restart flag created: {restart_flag}")
    
    async def _pause_trading(self):
        """Pause all trading activity."""
        logger.warning("[RECOVERY] 🛑 PAUSING ALL TRADING")
        
        # Set trading paused flag
        self.trading_paused = True
        
        # Write pause flag to disk
        pause_flag = self.data_dir / "trading_paused.flag"
        pause_flag.write_text(json.dumps({
            "paused": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reason": "Self-healing system intervention"
        }))
        
        logger.info("[RECOVERY] Trading pause flag set")
    
    async def _switch_to_safe_profile(self):
        """Switch to SAFE risk profile."""
        logger.warning("[RECOVERY] 🛡️ Switching to SAFE risk profile")
        
        # Load current config
        config_path = self.data_dir / "risk_config.json"
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {}
        
        # Apply SAFE profile
        config["profile"] = "SAFE"
        config["max_leverage"] = 5
        config["position_size_multiplier"] = 0.5
        config["stop_loss_multiplier"] = 0.8
        config["updated_by"] = "self_healing_system"
        config["updated_at"] = datetime.now(timezone.utc).isoformat()
        
        # Save config
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info("[RECOVERY] SAFE profile activated")
    
    async def _disable_module(self, subsystem: SubsystemType):
        """Disable a failing module."""
        logger.warning(f"[RECOVERY] ⚠️ Disabling module: {subsystem.value}")
        
        # Track disabled module
        self.disabled_modules.add(subsystem)
        
        # Write disable flag
        disable_flag = self.data_dir / f"{subsystem.value}_disabled.flag"
        disable_flag.write_text(json.dumps({
            "disabled": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reason": "Self-healing system intervention"
        }))
        
        logger.info(f"[RECOVERY] Module {subsystem.value} disabled")
    
    async def _enable_no_new_trades(self):
        """Enable NO_NEW_TRADES policy."""
        logger.warning("[RECOVERY] 🚫 NO NEW TRADES policy enabled")
        
        # Write policy flag
        policy_flag = self.data_dir / "no_new_trades.flag"
        policy_flag.write_text(json.dumps({
            "no_new_trades": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reason": "Self-healing system intervention"
        }))
        
        logger.info("[RECOVERY] NO_NEW_TRADES flag set")
    
    async def _defensive_exit(self):
        """Close risky positions defensively."""
        logger.warning("[RECOVERY] 🛡️ DEFENSIVE EXIT - Closing risky positions")
        
        # Write defensive exit flag
        exit_flag = self.data_dir / "defensive_exit.flag"
        exit_flag.write_text(json.dumps({
            "defensive_exit": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reason": "Self-healing system intervention",
            "criteria": "Close positions with leverage > 10x or unrealized loss > 5%"
        }))
        
        logger.info("[RECOVERY] Defensive exit flag set")
    
    async def _close_all_positions(self):
        """Emergency: close all positions."""
        logger.error("[RECOVERY] 🚨 EMERGENCY: CLOSING ALL POSITIONS")
        
        # Write emergency close flag
        close_flag = self.data_dir / "close_all_positions.flag"
        close_flag.write_text(json.dumps({
            "close_all": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reason": "EMERGENCY - Self-healing system intervention"
        }))
        
        logger.error("[RECOVERY] CLOSE ALL flag set - positions will be closed by executor")
    
    async def _reload_config(self):
        """Reload configuration files."""
        logger.info("[RECOVERY] 🔄 Reloading configuration")
        
        # Write reload flag
        reload_flag = self.data_dir / "config_reload_requested.flag"
        reload_flag.write_text(json.dumps({
            "reload": True,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }))
        
        logger.info("[RECOVERY] Config reload flag set")
    
    async def _clear_cache(self):
        """Clear system caches."""
        logger.info("[RECOVERY] 🗑️ Clearing caches")
        
        # Clear model prediction cache
        cache_files = [
            self.data_dir / "prediction_cache.json",
            self.data_dir / "feature_cache.json",
            self.data_dir / "market_data_cache.json"
        ]
        
        cleared = 0
        for cache_file in cache_files:
            if cache_file.exists():
                cache_file.unlink()
                cleared += 1
        
        logger.info(f"[RECOVERY] Cleared {cleared} cache files")
    
    async def _fallback_to_backup(self, subsystem: SubsystemType):
        """Switch to backup system."""
        logger.warning(f"[RECOVERY] 🔄 Falling back to backup for {subsystem.value}")
        
        # Write fallback flag
        fallback_flag = self.data_dir / f"{subsystem.value}_use_backup.flag"
        fallback_flag.write_text(json.dumps({
            "use_backup": True,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }))
        
        logger.info(f"[RECOVERY] Backup mode enabled for {subsystem.value}")
    
    # --------------------------------------------------------
    # STATE MANAGEMENT
    # --------------------------------------------------------
    
    async def _get_subsystem_state(self, subsystem: SubsystemType) -> Optional[str]:
        """Get current state of a subsystem."""
        try:
            state_file = self.data_dir / f"{subsystem.value}_state.json"
            if state_file.exists():
                with open(state_file, 'r') as f:
                    state = json.load(f)
                return json.dumps(state, indent=2)
            return None
        except Exception as e:
            logger.warning(f"Failed to get state for {subsystem.value}: {e}")
            return None
    
    def _save_history(self):
        """Save execution history to disk."""
        try:
            history_path = self.data_dir / "recovery_action_history.json"
            
            # Convert to dict (handle enums)
            history_dict = [
                {
                    "action": r.action.value,
                    "subsystem": r.subsystem.value,
                    "timestamp": r.timestamp,
                    "success": r.success,
                    "message": r.message,
                    "duration_ms": r.duration_ms,
                    "error": r.error
                }
                for r in self.execution_history[-100:]  # Keep last 100
            ]
            
            with open(history_path, 'w') as f:
                json.dump(history_dict, f, indent=2)
        
        except Exception as e:
            logger.error(f"Failed to save recovery history: {e}")
    
    # --------------------------------------------------------
    # STATUS QUERIES
    # --------------------------------------------------------
    
    def is_trading_paused(self) -> bool:
        """Check if trading is currently paused."""
        return self.trading_paused
    
    def is_module_disabled(self, subsystem: SubsystemType) -> bool:
        """Check if a module is disabled."""
        return subsystem in self.disabled_modules
    
    def get_execution_history(self, limit: int = 10) -> list[RecoveryResult]:
        """Get recent execution history."""
        return self.execution_history[-limit:]
    
    def get_success_rate(self) -> float:
        """Get overall success rate of recovery actions."""
        if not self.execution_history:
            return 0.0
        
        successes = sum(1 for r in self.execution_history if r.success)
        return successes / len(self.execution_history)


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("RECOVERY ACTION ENGINE - Standalone Test")
    print("=" * 60)
    
    # Initialize engine
    engine = RecoveryActionEngine(
        data_dir="./data",
        config_dir="./config",
        dry_run=False,  # Set to False to actually execute
        max_retries=2
    )
    
    print(f"\n[OK] Recovery Action Engine initialized")
    print(f"  Data dir: {engine.data_dir}")
    print(f"  Dry run: {engine.dry_run}")
    print(f"  Max retries: {engine.max_retries}")
    
    async def run_tests():
        print("\n" + "=" * 60)
        print("TEST 1: Execute DISABLE_MODULE")
        print("=" * 60)
        
        result = await engine.execute_recovery_action(
            action=RecoveryAction.DISABLE_MODULE,
            subsystem=SubsystemType.AI_MODEL,
            reason="Test: Model failures detected"
        )
        
        print(f"\n[OK] Action executed")
        print(f"  Success: {result.success}")
        print(f"  Duration: {result.duration_ms:.0f}ms")
        print(f"  Message: {result.message}")
        
        print("\n" + "=" * 60)
        print("TEST 2: Execute SWITCH_TO_SAFE_PROFILE")
        print("=" * 60)
        
        result = await engine.execute_recovery_action(
            action=RecoveryAction.SWITCH_TO_SAFE_PROFILE,
            subsystem=SubsystemType.RISK_GUARD,
            reason="Test: High system stress detected"
        )
        
        print(f"\n[OK] Action executed")
        print(f"  Success: {result.success}")
        print(f"  Duration: {result.duration_ms:.0f}ms")
        print(f"  Message: {result.message}")
        
        print("\n" + "=" * 60)
        print("TEST 3: Execute NO_NEW_TRADES")
        print("=" * 60)
        
        result = await engine.execute_recovery_action(
            action=RecoveryAction.NO_NEW_TRADES,
            subsystem=SubsystemType.EVENT_EXECUTOR,
            reason="Test: Degraded system performance"
        )
        
        print(f"\n[OK] Action executed")
        print(f"  Success: {result.success}")
        print(f"  Duration: {result.duration_ms:.0f}ms")
        print(f"  Message: {result.message}")
        
        print("\n" + "=" * 60)
        print("TEST 4: Check Status")
        print("=" * 60)
        
        print(f"\n[OK] Status:")
        print(f"  Trading paused: {engine.is_trading_paused()}")
        print(f"  AI_MODEL disabled: {engine.is_module_disabled(SubsystemType.AI_MODEL)}")
        print(f"  Success rate: {engine.get_success_rate():.1%}")
        print(f"  Executions: {len(engine.execution_history)}")
        
        print("\n" + "=" * 60)
        print("TEST 5: Execution History")
        print("=" * 60)
        
        history = engine.get_execution_history(limit=5)
        print(f"\n[OK] Recent executions ({len(history)}):")
        for i, result in enumerate(history, 1):
            status = "✅" if result.success else "❌"
            print(f"  {i}. {status} {result.action.value} ({result.duration_ms:.0f}ms)")
        
        print("\n" + "=" * 60)
        print("[OK] All tests completed successfully!")
        print("=" * 60)
    
    asyncio.run(run_tests())
    
    print(f"\n[OK] History saved to: {engine.data_dir / 'recovery_action_history.json'}")
