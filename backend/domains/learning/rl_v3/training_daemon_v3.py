"""
RL v3 Training Daemon - Automated background PPO training service.

Periodically trains the PPO agent based on PolicyStore schedule configuration.
Publishes training events to EventBus v2 and records metrics.

Author: Quantum Trader AI Team
Date: December 2, 2025
Version: 3.0 (Production)
"""

import asyncio
import traceback
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional

import structlog

from backend.domains.learning.rl_v3.rl_manager_v3 import RLv3Manager
from backend.domains.learning.rl_v3.metrics_v3 import RLv3MetricsStore
from backend.core.event_bus import EventBus
from backend.core.policy_store import PolicyStore


logger = structlog.get_logger(__name__)


# Default configuration
DEFAULT_CONFIG = {
    "enabled": True,
    "interval_minutes": 30,
    "episodes_per_run": 2,
}


class RLv3TrainingDaemon:
    """
    Background daemon for periodic RL v3 PPO training.
    
    Features:
    - Automatic scheduled training based on PolicyStore config
    - Live reload of config without restart
    - EventBus integration (publishes training events)
    - Metrics tracking via RLv3MetricsStore
    - Structured logging with run IDs
    """
    
    def __init__(
        self,
        rl_manager: RLv3Manager,
        event_bus: Optional[EventBus] = None,
        policy_store: Optional[PolicyStore] = None,
        logger_instance: Any = None
    ):
        """
        Initialize training daemon.
        
        Args:
            rl_manager: RLv3Manager instance to train
            event_bus: EventBus v2 for publishing events (optional)
            policy_store: PolicyStore v2 for config (optional)
            logger_instance: Structured logger (optional)
        """
        self.rl_manager = rl_manager
        self.event_bus = event_bus
        self.policy_store = policy_store
        self.logger = logger_instance or logger
        
        # State
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_run: Optional[datetime] = None
        self._in_progress = False
        self._config_refresh_counter = 0
        
        # Metrics
        self.metrics = RLv3MetricsStore.instance()
        self.total_runs = 0
        self.successful_runs = 0
        self.failed_runs = 0
        
        # Load initial config
        self._load_config()
        
        self.logger.info(
            "[RLv3][TRAINING] Daemon initialized",
            enabled=self.config["enabled"],
            interval_minutes=self.config["interval_minutes"],
            episodes_per_run=self.config["episodes_per_run"]
        )
    
    def _load_config(self):
        """Load configuration from PolicyStore or use defaults."""
        self.config = DEFAULT_CONFIG.copy()
        
        if self.policy_store:
            try:
                self.config["enabled"] = self.policy_store.get(
                    "rl_v3.training.enabled", 
                    DEFAULT_CONFIG["enabled"]
                )
                self.config["interval_minutes"] = int(self.policy_store.get(
                    "rl_v3.training.interval_minutes", 
                    DEFAULT_CONFIG["interval_minutes"]
                ))
                self.config["episodes_per_run"] = int(self.policy_store.get(
                    "rl_v3.training.episodes_per_run", 
                    DEFAULT_CONFIG["episodes_per_run"]
                ))
                
                self.logger.debug(
                    "[RLv3][TRAINING] Config loaded from PolicyStore",
                    config=self.config
                )
            except Exception as e:
                self.logger.warning(
                    "[RLv3][TRAINING] Failed to load PolicyStore config, using defaults",
                    error=str(e)
                )
    
    def _refresh_config(self):
        """Refresh configuration from PolicyStore (live reload)."""
        if not self.policy_store:
            return
        
        old_config = self.config.copy()
        self._load_config()
        
        if old_config != self.config:
            self.logger.info(
                "[RLv3][TRAINING] Config updated",
                old=old_config,
                new=self.config
            )
    
    async def start(self):
        """Start the training daemon."""
        if self._running:
            self.logger.warning("[RLv3][TRAINING] Daemon already running")
            return
        
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        
        self.logger.info(
            "[RLv3][TRAINING] Daemon started",
            enabled=self.config["enabled"],
            interval_minutes=self.config["interval_minutes"]
        )
    
    async def stop(self):
        """Stop the training daemon."""
        if not self._running:
            return
        
        self._running = False
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        
        self.logger.info("[RLv3][TRAINING] Daemon stopped")
    
    async def _run_loop(self):
        """Main daemon loop - checks schedule and runs training."""
        self.logger.debug("[RLv3][TRAINING] Loop started")
        
        while self._running:
            try:
                # Refresh config every 10 iterations (~100 seconds)
                self._config_refresh_counter += 1
                if self._config_refresh_counter >= 10:
                    self._refresh_config()
                    self._config_refresh_counter = 0
                
                # Check if training is enabled
                if not self.config["enabled"]:
                    await asyncio.sleep(30)
                    continue
                
                # Check if enough time has passed since last run
                now = datetime.now(timezone.utc)
                
                if self._last_run is not None:
                    interval = timedelta(minutes=self.config["interval_minutes"])
                    time_since_last = now - self._last_run
                    
                    if time_since_last < interval:
                        # Sleep for a short time to avoid busy-waiting
                        await asyncio.sleep(10)
                        continue
                
                # Check if training already in progress
                if self._in_progress:
                    await asyncio.sleep(10)
                    continue
                
                # Run training cycle
                await self._run_training_cycle()
                
            except asyncio.CancelledError:
                self.logger.debug("[RLv3][TRAINING] Loop cancelled")
                break
            except Exception as e:
                self.logger.error(
                    "[RLv3][TRAINING] Loop error",
                    error=str(e),
                    exc_info=True
                )
                await asyncio.sleep(30)
    
    async def run_once(self) -> Dict[str, Any]:
        """
        Manually trigger one training run (for testing).
        
        Returns:
            Training metrics dict
        """
        return await self._run_training_cycle()
    
    async def _run_training_cycle(self) -> Dict[str, Any]:
        """Execute a single training cycle."""
        self._in_progress = True
        run_id = str(uuid.uuid4())[:8]
        start_time = datetime.now(timezone.utc)
        
        try:
            episodes = self.config["episodes_per_run"]
            
            self.logger.info(
                f"[RLv3][TRAINING][RUN_ID={run_id}] Starting scheduled run",
                episodes=episodes,
                timestamp=start_time.isoformat()
            )
            
            # Publish training.started event
            if self.event_bus:
                await self._publish_event(
                    "rl_v3.training.started",
                    {
                        "run_id": run_id,
                        "episodes": episodes,
                        "timestamp": start_time.isoformat()
                    }
                )
            
            # Run training in executor to avoid blocking event loop
            loop = asyncio.get_running_loop()
            
            # Log progress during training
            self.logger.info(
                f"[RLv3][TRAINING][RUN_ID={run_id}][episode=0/{episodes}] Training started"
            )
            
            training_metrics = await loop.run_in_executor(
                None,
                self.rl_manager.train,
                episodes
            )
            
            # Save model after training
            await loop.run_in_executor(None, self.rl_manager.save)
            
            # Calculate metrics
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            
            avg_reward = training_metrics.get("avg_reward", 0.0)
            final_reward = training_metrics.get("final_reward", 0.0)
            avg_policy_loss = (
                sum(training_metrics.get("policy_losses", [0])) / 
                max(len(training_metrics.get("policy_losses", [1])), 1)
            )
            avg_value_loss = (
                sum(training_metrics.get("value_losses", [0])) / 
                max(len(training_metrics.get("value_losses", [1])), 1)
            )
            
            # Record to metrics store
            run_data = {
                "run_id": run_id,
                "timestamp": start_time.isoformat(),
                "episodes": episodes,
                "duration_seconds": duration,
                "success": True,
                "error": None,
                "avg_reward": avg_reward,
                "final_reward": final_reward,
                "avg_policy_loss": avg_policy_loss,
                "avg_value_loss": avg_value_loss
            }
            
            self.metrics.record_training_run(run_data)
            
            # Update counters
            self.total_runs += 1
            self.successful_runs += 1
            self._last_run = end_time
            
            self.logger.info(
                f"[RLv3][TRAINING][RUN_ID={run_id}][episode={episodes}/{episodes}] Training completed",
                duration_seconds=duration,
                avg_reward=avg_reward,
                final_reward=final_reward
            )
            
            # Publish training.completed event
            if self.event_bus:
                await self._publish_event(
                    "rl_v3.training.completed",
                    {
                        "run_id": run_id,
                        "success": True,
                        "episodes": episodes,
                        "duration_seconds": duration,
                        "avg_reward": avg_reward,
                        "final_reward": final_reward,
                        "timestamp": end_time.isoformat()
                    }
                )
            
            return run_data
            
        except Exception as e:
            # Record failure
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            
            error_msg = str(e)
            error_trace = traceback.format_exc()
            
            run_data = {
                "run_id": run_id,
                "timestamp": start_time.isoformat(),
                "episodes": self.config["episodes_per_run"],
                "duration_seconds": duration,
                "success": False,
                "error": error_msg,
                "traceback": error_trace
            }
            
            self.metrics.record_training_run(run_data)
            
            # Update counters
            self.total_runs += 1
            self.failed_runs += 1
            
            self.logger.error(
                f"[RLv3][TRAINING][RUN_ID={run_id}] Training failed",
                error=error_msg,
                duration_seconds=duration,
                exc_info=True
            )
            
            # Publish training.completed event (with failure)
            if self.event_bus:
                await self._publish_event(
                    "rl_v3.training.completed",
                    {
                        "run_id": run_id,
                        "success": False,
                        "error": error_msg,
                        "duration_seconds": duration,
                        "timestamp": end_time.isoformat()
                    }
                )
            
            return run_data
        
        finally:
            self._in_progress = False
    
    async def _publish_event(self, event_type: str, payload: Dict[str, Any]):
        """Publish event to EventBus."""
        try:
            await self.event_bus.publish(
                event_type=event_type,
                payload=payload,
                trace_id=payload.get("run_id", "")
            )
            
            self.logger.debug(
                "[RLv3][TRAINING] Event published",
                event_type=event_type,
                run_id=payload.get("run_id")
            )
        except Exception as e:
            self.logger.error(
                "[RLv3][TRAINING] Failed to publish event",
                event_type=event_type,
                error=str(e)
            )
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current daemon status.
        
        Returns:
            Status dict with config and statistics
        """
        return {
            "running": self._running,
            "enabled": self.config["enabled"],
            "interval_minutes": self.config["interval_minutes"],
            "episodes_per_run": self.config["episodes_per_run"],
            "total_runs": self.total_runs,
            "successful_runs": self.successful_runs,
            "failed_runs": self.failed_runs,
            "success_rate": (
                self.successful_runs / self.total_runs 
                if self.total_runs > 0 else 0.0
            ),
            "last_run_at": (
                self._last_run.isoformat() 
                if self._last_run else None
            ),
            "in_progress": self._in_progress
        }
