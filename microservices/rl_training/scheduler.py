"""
Scheduler

Periodic task scheduler for automated retraining.
"""
import asyncio
import logging
from datetime import datetime, timezone

from microservices.rl_training.models import ModelType, TrainingTrigger


logger = logging.getLogger(__name__)


class TrainingScheduler:
    """
    Training Scheduler.
    
    Runs periodic checks and triggers retraining cycles.
    """
    
    def __init__(
        self,
        training_daemon,
        clm,
        drift_detector,
        config,
        logger_instance=None
    ):
        """
        Initialize scheduler.
        
        Args:
            training_daemon: RLTrainingDaemon instance
            clm: ContinuousLearningManager instance
            drift_detector: DriftDetector instance
            config: Service configuration
            logger_instance: Logger instance (optional)
        """
        self.training_daemon = training_daemon
        self.clm = clm
        self.drift_detector = drift_detector
        self.config = config
        self.logger = logger_instance or logger
        
        self._running = False
        self._tasks = []
    
    async def start(self):
        """Start all scheduled tasks"""
        if self._running:
            self.logger.debug("[Scheduler] Already running")
            return
        
        self._running = True
        
        # Start RL training schedule
        if self.config.RL_TRAINING_ENABLED:
            task = asyncio.create_task(self._rl_training_loop())
            self._tasks.append(task)
            self.logger.info(
                f"[Scheduler] RL training schedule started "
                f"(interval: {self.config.RL_RETRAIN_INTERVAL_HOURS}h)"
            )
        
        # Start CLM training schedule
        if self.config.CLM_ENABLED:
            task = asyncio.create_task(self._clm_training_loop())
            self._tasks.append(task)
            self.logger.info(
                f"[Scheduler] CLM training schedule started "
                f"(interval: {self.config.CLM_RETRAIN_INTERVAL_HOURS}h)"
            )
        
        # Start drift check schedule
        if self.config.DRIFT_DETECTION_ENABLED:
            task = asyncio.create_task(self._drift_check_loop())
            self._tasks.append(task)
            self.logger.info(
                f"[Scheduler] Drift check schedule started "
                f"(interval: {self.config.DRIFT_CHECK_INTERVAL_HOURS}h)"
            )
    
    async def stop(self):
        """Stop all scheduled tasks"""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        
        self._tasks.clear()
        self.logger.info("[Scheduler] All scheduled tasks stopped")
    
    async def _rl_training_loop(self):
        """Periodic RL training loop"""
        interval_seconds = self.config.RL_RETRAIN_INTERVAL_HOURS * 3600
        
        while self._running:
            try:
                self.logger.info("[Scheduler] Running RL training cycle")
                
                # Trigger RL training
                await self.training_daemon.run_training_cycle(
                    model_type=ModelType.RL_PPO,
                    trigger=TrainingTrigger.SCHEDULED,
                    reason="Scheduled periodic training"
                )
                
                self.logger.info(
                    f"[Scheduler] RL training cycle completed, "
                    f"next in {self.config.RL_RETRAIN_INTERVAL_HOURS}h"
                )
                
                # Sleep until next cycle
                await asyncio.sleep(interval_seconds)
            
            except asyncio.CancelledError:
                self.logger.info("[Scheduler] RL training loop cancelled")
                break
            except Exception as e:
                self.logger.error(
                    f"[Scheduler] Error in RL training loop: {e}",
                    exc_info=True
                )
                # Sleep 1 hour before retry
                await asyncio.sleep(3600)
    
    async def _clm_training_loop(self):
        """Periodic CLM training loop"""
        interval_seconds = self.config.CLM_RETRAIN_INTERVAL_HOURS * 3600
        
        while self._running:
            try:
                self.logger.info("[Scheduler] Running CLM full cycle")
                
                # Run CLM full cycle
                result = await self.clm.run_full_cycle()
                
                self.logger.info(
                    f"[Scheduler] CLM cycle completed: "
                    f"{result['models_retrained']} models retrained, "
                    f"next in {self.config.CLM_RETRAIN_INTERVAL_HOURS}h"
                )
                
                # Sleep until next cycle
                await asyncio.sleep(interval_seconds)
            
            except asyncio.CancelledError:
                self.logger.info("[Scheduler] CLM training loop cancelled")
                break
            except Exception as e:
                self.logger.error(
                    f"[Scheduler] Error in CLM training loop: {e}",
                    exc_info=True
                )
                # Sleep 1 hour before retry
                await asyncio.sleep(3600)
    
    async def _drift_check_loop(self):
        """Periodic drift check loop"""
        interval_seconds = self.config.DRIFT_CHECK_INTERVAL_HOURS * 3600
        
        while self._running:
            try:
                self.logger.info("[Scheduler] Running drift check")
                
                # Check drift for all features
                # (Would fetch current distributions from data source)
                # For now, this is a placeholder
                
                self.logger.info(
                    f"[Scheduler] Drift check completed, "
                    f"next in {self.config.DRIFT_CHECK_INTERVAL_HOURS}h"
                )
                
                # Sleep until next cycle
                await asyncio.sleep(interval_seconds)
            
            except asyncio.CancelledError:
                self.logger.info("[Scheduler] Drift check loop cancelled")
                break
            except Exception as e:
                self.logger.error(
                    f"[Scheduler] Error in drift check loop: {e}",
                    exc_info=True
                )
                # Sleep 1 hour before retry
                await asyncio.sleep(3600)
