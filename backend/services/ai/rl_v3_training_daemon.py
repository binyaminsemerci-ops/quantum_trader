"""
RL v3 Training Daemon - Background service for periodic PPO training.

This daemon periodically trains the RLv3Manager (PPO agent) in the background,
saves updated model weights, and logs training metrics.

Can be controlled via PolicyStore v2 for live configuration updates.

Author: Quantum Trader AI Team
Date: December 2, 2025
Version: 3.0
"""

import asyncio
import traceback
from datetime import datetime, timezone, timedelta
from typing import Optional, TYPE_CHECKING

from backend.domains.learning.rl_v3.rl_manager_v3 import RLv3Manager
from backend.domains.learning.rl_v3.training_config_v3 import (
    RLv3TrainingConfig,
    DEFAULT_TRAINING_CONFIG,
    load_training_config_from_policy_store
)
from backend.domains.learning.rl_v3.metrics_v3 import RLv3MetricsStore

if TYPE_CHECKING:
    from backend.domains.policy_store_v2 import PolicyStoreV2


class RLv3TrainingDaemon:
    """
    Background daemon for periodic RL v3 PPO training.
    
    Runs training cycles at configured intervals, saves model weights,
    and records training metrics.
    
    Can be dynamically configured via PolicyStore v2.
    """
    
    def __init__(
        self,
        rl_manager: RLv3Manager,
        config: Optional[RLv3TrainingConfig] = None,
        logger=None,
        policy_store: Optional["PolicyStoreV2"] = None
    ):
        """
        Initialize training daemon.
        
        Args:
            rl_manager: RLv3Manager instance to train
            config: Training configuration (uses default if None)
            logger: Structured logger (optional)
            policy_store: PolicyStore v2 for live config updates (optional)
        """
        self.rl_manager = rl_manager
        self.logger = logger
        self.policy_store = policy_store
        
        # Load initial config
        if config is None and policy_store is not None:
            self.config = load_training_config_from_policy_store(policy_store)
            if self.logger:
                self.logger.info(
                    "[RL v3 Training Daemon] Config loaded from PolicyStore",
                    enabled=self.config.schedule.enabled,
                    interval_minutes=self.config.schedule.interval_minutes,
                    episodes_per_run=self.config.schedule.episodes_per_run
                )
        else:
            self.config = config or DEFAULT_TRAINING_CONFIG
        
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_run: Optional[datetime] = None
        self._in_progress = False
        self._config_check_counter = 0
    
    async def start(self):
        """Start the training daemon."""
        if self._running:
            if self.logger:
                self.logger.debug("[RL v3 Training Daemon] Already running")
            return
        
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        
        if self.logger:
            self.logger.info(
                "[RL v3 Training Daemon] Started",
                enabled=self.config.schedule.enabled,
                interval_minutes=self.config.schedule.interval_minutes,
                episodes_per_run=self.config.schedule.episodes_per_run
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
        
        if self.logger:
            self.logger.info("[RL v3 Training Daemon] Stopped")
    
    async def _run_loop(self):
        """Main daemon loop - checks schedule and runs training."""
        if self.logger:
            self.logger.debug("[RL v3 Training Daemon] Loop started")
        
        while self._running:
            try:
                # Refresh config from PolicyStore every ~10 iterations (~100 seconds)
                self._config_check_counter += 1
                if self.policy_store and self._config_check_counter >= 10:
                    self._refresh_config()
                    self._config_check_counter = 0
                
                # Check if training is enabled
                if not self.config.schedule.enabled:
                    await asyncio.sleep(30)
                    continue
                
                # Check if enough time has passed since last run
                now = datetime.now(timezone.utc)
                
                if self._last_run is not None:
                    interval = timedelta(minutes=self.config.schedule.interval_minutes)
                    if now - self._last_run < interval:
                        await asyncio.sleep(10)
                        continue
                
                # Check if training already in progress
                if self._in_progress:
                    await asyncio.sleep(10)
                    continue
                
                # Run training cycle
                await self._run_training_cycle()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.logger:
                    self.logger.error(
                        "[RL v3 Training Daemon] Loop error",
                        error=str(e),
                        exc_info=True
                    )
                await asyncio.sleep(30)
    
    def _refresh_config(self):
        """Refresh configuration from PolicyStore."""
        try:
            new_config = load_training_config_from_policy_store(self.policy_store)
            
            # Check if config changed
            old_enabled = self.config.schedule.enabled
            old_interval = self.config.schedule.interval_minutes
            old_episodes = self.config.schedule.episodes_per_run
            
            changed = (
                new_config.schedule.enabled != old_enabled or
                new_config.schedule.interval_minutes != old_interval or
                new_config.schedule.episodes_per_run != old_episodes
            )
            
            if changed:
                self.config = new_config
                if self.logger:
                    self.logger.info(
                        "[RL v3 Training Daemon] Config updated from PolicyStore",
                        enabled=self.config.schedule.enabled,
                        interval_minutes=self.config.schedule.interval_minutes,
                        episodes_per_run=self.config.schedule.episodes_per_run
                    )
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "[RL v3 Training Daemon] Failed to refresh config",
                    error=str(e)
                )
    
    async def _run_training_cycle(self):
        """Execute a single training cycle."""
        self._in_progress = True
        start_time = datetime.now(timezone.utc)
        
        try:
            episodes = self.config.schedule.episodes_per_run
            
            if self.logger:
                self.logger.info(
                    "[RL v3 Training Daemon] Starting training cycle",
                    episodes=episodes,
                    timestamp=start_time.isoformat()
                )
            
            # Run training in executor to avoid blocking event loop
            loop = asyncio.get_running_loop()
            metrics = await loop.run_in_executor(
                None,
                self.rl_manager.train,
                episodes
            )
            
            # Save model if configured
            if self.config.schedule.save_after_each_run:
                await loop.run_in_executor(None, self.rl_manager.save)
            
            # Calculate duration
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            
            # Record training run in metrics store
            metrics_store = RLv3MetricsStore.instance()
            metrics_store.record_training_run({
                "timestamp": start_time.isoformat(),
                "episodes": episodes,
                "duration_seconds": duration,
                "success": True,
                "error": None,
                "avg_reward": metrics.get("avg_reward", 0.0),
                "final_reward": metrics.get("final_reward", 0.0),
                "avg_policy_loss": (
                    sum(metrics.get("policy_losses", [0])) / 
                    max(len(metrics.get("policy_losses", [1])), 1)
                ),
                "avg_value_loss": (
                    sum(metrics.get("value_losses", [0])) / 
                    max(len(metrics.get("value_losses", [1])), 1)
                )
            })
            
            # Update last run time
            self._last_run = end_time
            
            if self.logger:
                self.logger.info(
                    "[RL v3 Training Daemon] Training cycle completed",
                    duration_seconds=duration,
                    avg_reward=metrics.get("avg_reward", 0.0),
                    final_reward=metrics.get("final_reward", 0.0)
                )
            
        except Exception as e:
            # Record failure
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            
            error_msg = str(e)
            error_trace = traceback.format_exc()
            
            metrics_store = RLv3MetricsStore.instance()
            metrics_store.record_training_run({
                "timestamp": start_time.isoformat(),
                "episodes": self.config.schedule.episodes_per_run,
                "duration_seconds": duration,
                "success": False,
                "error": error_msg,
                "traceback": error_trace
            })
            
            if self.logger:
                self.logger.error(
                    "[RL v3 Training Daemon] Training cycle failed",
                    error=error_msg,
                    duration_seconds=duration,
                    exc_info=True
                )
        
        finally:
            self._in_progress = False
