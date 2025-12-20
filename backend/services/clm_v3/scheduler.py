"""
CLM v3 Scheduler - Training trigger logic.

Determines when to train models based on:
- Periodic schedules (daily, weekly)
- Drift detection events
- Performance degradation
- Manual triggers
- Data volume thresholds
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from backend.services.clm_v3.models import ModelType, TriggerReason, TrainingJob
from backend.services.clm_v3.storage import ModelRegistryV3

logger = logging.getLogger(__name__)


class TrainingScheduler:
    """
    Training Scheduler - Determines when models should be retrained.
    
    Features:
    - Periodic scheduling (configurable intervals)
    - Event-driven triggers (drift, performance)
    - Data volume triggers
    - Manual triggers
    """
    
    def __init__(
        self,
        registry: ModelRegistryV3,
        config: Optional[Dict] = None,
    ):
        """
        Initialize Training Scheduler.
        
        Args:
            registry: ModelRegistryV3 instance
            config: Scheduler configuration
        """
        self.registry = registry
        
        # Merge provided config with defaults (deep merge for nested dicts)
        default_config = self._default_config()
        if config:
            # Merge top-level keys
            for key, value in config.items():
                if key in default_config and isinstance(default_config[key], dict) and isinstance(value, dict):
                    # Deep merge for nested dicts like "periodic_training"
                    default_config[key].update(value)
                else:
                    default_config[key] = value
        self.config = default_config
        
        # State
        self.last_training: Dict[str, datetime] = {}  # {model_id: last_trained_at}
        self._running = False
        self._task: Optional[asyncio.Task] = None
        
        logger.info("[CLM v3 Scheduler] Initialized")
    
    @staticmethod
    def _default_config() -> Dict:
        """Default scheduler configuration."""
        return {
            "enabled": True,
            "check_interval_minutes": 30,  # How often to check if training needed
            
            # Periodic training
            "periodic_training": {
                "enabled": True,
                "xgboost_interval_hours": 168,  # Weekly (7 days)
                "lightgbm_interval_hours": 168,
                "nhits_interval_hours": 336,  # Bi-weekly (14 days)
                "patchtst_interval_hours": 336,
                "rl_v3_interval_hours": 24,  # Daily
            },
            
            # Data volume triggers
            "data_volume_trigger": {
                "enabled": True,
                "min_new_trades": 100,  # Min new trades since last training
            },
            
            # Drift triggers (handled by DriftDetectionManager events)
            "drift_trigger": {
                "enabled": True,
                "auto_train_on_drift": True,
            },
            
            # Performance triggers
            "performance_trigger": {
                "enabled": True,
                "auto_train_on_degradation": True,
                "sharpe_threshold": 0.5,  # Train if Sharpe < this
            },
        }
    
    # ========================================================================
    # Periodic Scheduling
    # ========================================================================
    
    async def start(self):
        """Start scheduler background task."""
        if self._running:
            logger.warning("[CLM v3 Scheduler] Already running")
            return
        
        self._running = True
        self._task = asyncio.create_task(self._scheduler_loop())
        logger.info("[CLM v3 Scheduler] Started")
    
    async def stop(self):
        """Stop scheduler background task."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("[CLM v3 Scheduler] Stopped")
    
    async def _scheduler_loop(self):
        """Main scheduler loop - checks for training needs periodically."""
        interval_seconds = self.config["check_interval_minutes"] * 60
        logger.info(f"[CLM v3 Scheduler] Loop started - will check every {interval_seconds}s ({interval_seconds/60:.1f} min)")
        
        while self._running:
            try:
                logger.info(f"[CLM v3 Scheduler] 🔍 Running periodic training check...")
                await self._check_periodic_training()
                logger.info(f"[CLM v3 Scheduler] ✅ Check complete - sleeping {interval_seconds}s")
            except Exception as e:
                logger.error(f"[CLM v3 Scheduler] Error in scheduler loop: {e}", exc_info=True)
            
            await asyncio.sleep(interval_seconds)
    
    async def _check_periodic_training(self):
        """Check if any models need periodic retraining."""
        if not self.config["periodic_training"]["enabled"]:
            return
        
        now = datetime.utcnow()
        periodic_config = self.config["periodic_training"]
        
        logger.debug(f"[CLM v3 Scheduler] 🔍 Checking for models needing periodic training...")
        
        # Check each model type
        for model_type in ModelType:
            if model_type == ModelType.OTHER:
                continue
            
            # Get interval for this model type
            interval_key = f"{model_type.value}_interval_hours"
            interval_hours = periodic_config.get(interval_key, 168)  # Default: weekly
            
            # Check if training needed
            model_id = f"{model_type.value}_main"  # Generic model ID
            last_trained = self.last_training.get(model_id)
            
            if last_trained is None:
                # Never trained - check registry
                prod_model = self.registry.get_production_model(model_id)
                if prod_model:
                    last_trained = prod_model.created_at
            
            if last_trained is None:
                # Still no training history - schedule training
                logger.info(
                    f"[CLM v3 Scheduler] No training history for {model_id} - scheduling initial training"
                )
                await self.trigger_training(
                    model_type=model_type,
                    trigger_reason=TriggerReason.PERIODIC,
                    triggered_by="scheduler_initial",
                )
                self.last_training[model_id] = now
                continue
            
            # Check if enough time has passed
            time_since_training = (now - last_trained).total_seconds() / 3600
            if time_since_training >= interval_hours:
                logger.info(
                    f"[CLM v3 Scheduler] Periodic training due for {model_id} "
                    f"(last trained {time_since_training:.1f}h ago, interval={interval_hours}h)"
                )
                await self.trigger_training(
                    model_type=model_type,
                    trigger_reason=TriggerReason.PERIODIC,
                    triggered_by="scheduler_periodic",
                )
                self.last_training[model_id] = now
    
    # ========================================================================
    # Event-Driven Triggers
    # ========================================================================
    
    async def handle_drift_detected(
        self,
        model_id: str,
        model_type: ModelType,
        drift_score: float,
    ):
        """Handle drift detection event."""
        if not self.config["drift_trigger"]["enabled"]:
            logger.info(f"[CLM v3 Scheduler] Drift detected for {model_id}, but drift trigger disabled")
            return
        
        if not self.config["drift_trigger"]["auto_train_on_drift"]:
            logger.info(f"[CLM v3 Scheduler] Drift detected for {model_id}, but auto-train disabled")
            return
        
        logger.warning(
            f"[CLM v3 Scheduler] Drift detected for {model_id} (score={drift_score:.3f}) "
            f"- triggering retraining"
        )
        
        await self.trigger_training(
            model_type=model_type,
            trigger_reason=TriggerReason.DRIFT_DETECTED,
            triggered_by=f"drift_detector_{drift_score:.3f}",
        )
    
    async def handle_performance_degraded(
        self,
        model_id: str,
        model_type: ModelType,
        sharpe_ratio: float,
    ):
        """Handle performance degradation event."""
        if not self.config["performance_trigger"]["enabled"]:
            return
        
        threshold = self.config["performance_trigger"]["sharpe_threshold"]
        if sharpe_ratio >= threshold:
            return  # Performance is acceptable
        
        if not self.config["performance_trigger"]["auto_train_on_degradation"]:
            logger.info(
                f"[CLM v3 Scheduler] Performance degraded for {model_id} "
                f"(sharpe={sharpe_ratio:.3f}), but auto-train disabled"
            )
            return
        
        logger.warning(
            f"[CLM v3 Scheduler] Performance degraded for {model_id} "
            f"(sharpe={sharpe_ratio:.3f} < {threshold}) - triggering retraining"
        )
        
        await self.trigger_training(
            model_type=model_type,
            trigger_reason=TriggerReason.PERFORMANCE_DEGRADED,
            triggered_by=f"performance_monitor_{sharpe_ratio:.3f}",
        )
    
    async def handle_regime_change(
        self,
        new_regime: str,
        affected_models: List[str],
    ):
        """Handle market regime change event."""
        logger.info(
            f"[CLM v3 Scheduler] Market regime changed to {new_regime} "
            f"- scheduling retraining for {len(affected_models)} models"
        )
        
        # TODO: Implement regime-specific model retraining
        # For now, just log
        pass
    
    # ========================================================================
    # Manual Triggers
    # ========================================================================
    
    async def trigger_training(
        self,
        model_type: ModelType,
        trigger_reason: TriggerReason,
        triggered_by: str = "manual",
        symbol: Optional[str] = None,
        timeframe: str = "1h",
        dataset_span_days: int = 90,
        training_params: Optional[Dict] = None,
    ) -> TrainingJob:
        """
        Manually trigger a training job.
        
        Args:
            model_type: Type of model to train
            trigger_reason: Why training was triggered
            triggered_by: Who/what triggered it
            symbol: Specific symbol or None for multi-symbol
            timeframe: Timeframe to train on
            dataset_span_days: How many days of data
            training_params: Optional training parameters
        
        Returns:
            Created TrainingJob
        """
        job = TrainingJob(
            model_type=model_type,
            symbol=symbol,
            timeframe=timeframe,
            dataset_span_days=dataset_span_days,
            trigger_reason=trigger_reason,
            triggered_by=triggered_by,
            training_params=training_params or {},
            status="pending",
        )
        
        # Register in registry
        job = self.registry.register_training_job(job)
        
        logger.info(
            f"[CLM v3 Scheduler] Created training job {job.id} "
            f"(model={model_type.value}, trigger={trigger_reason.value}, by={triggered_by})"
        )
        
        return job
    
    # ========================================================================
    # Utilities
    # ========================================================================
    
    def get_next_training_times(self) -> Dict[str, datetime]:
        """Get estimated next training time for each model type."""
        now = datetime.utcnow()
        next_times = {}
        
        periodic_config = self.config["periodic_training"]
        
        for model_type in ModelType:
            if model_type == ModelType.OTHER:
                continue
            
            interval_key = f"{model_type.value}_interval_hours"
            interval_hours = periodic_config.get(interval_key, 168)
            
            model_id = f"{model_type.value}_main"
            last_trained = self.last_training.get(model_id)
            
            if last_trained:
                next_time = last_trained + timedelta(hours=interval_hours)
            else:
                next_time = now  # Train ASAP if never trained
            
            next_times[model_id] = next_time
        
        return next_times
    
    def get_status(self) -> Dict:
        """Get scheduler status."""
        return {
            "running": self._running,
            "config": self.config,
            "last_training": {
                model_id: ts.isoformat()
                for model_id, ts in self.last_training.items()
            },
            "next_training_times": {
                model_id: ts.isoformat()
                for model_id, ts in self.get_next_training_times().items()
            },
        }
