"""
Continuous Learning Manager (CLM)

Orchestrates retraining of supervised ML models (XGB, LGBM, etc.).
Delegates to TrainingDaemon for actual training.
"""
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List

from microservices.rl_training.models import (
    ModelType,
    TrainingTrigger,
    TrainingConfig,
)


logger = logging.getLogger(__name__)


class ContinuousLearningManager:
    """
    Continuous Learning Manager.
    
    Orchestrates periodic retraining of supervised ML models.
    Works alongside RLTrainingDaemon.
    """
    
    def __init__(
        self,
        training_daemon,
        config,
        logger_instance=None
    ):
        """
        Initialize CLM.
        
        Args:
            training_daemon: RLTrainingDaemon instance
            config: Service configuration
            logger_instance: Logger instance (optional)
        """
        self.training_daemon = training_daemon
        self.config = config
        self.logger = logger_instance or logger
        
        self._last_retrain_time: Dict[ModelType, datetime] = {}
        self._retrain_triggered: Dict[ModelType, List[str]] = {}
    
    async def check_if_retrain_needed(self) -> Dict[ModelType, bool]:
        """
        Check if any models need retraining.
        
        Returns:
            Dictionary mapping model types to whether they need retraining
        """
        now = datetime.now(timezone.utc)
        results = {}
        
        # Check time-based triggers
        for model_type in [ModelType.XGBOOST, ModelType.LIGHTGBM]:
            last_retrain = self._last_retrain_time.get(model_type)
            
            if last_retrain is None:
                # Never trained
                results[model_type] = True
                self._retrain_triggered.setdefault(model_type, []).append(
                    "initial_training"
                )
            else:
                hours_since_retrain = (now - last_retrain).total_seconds() / 3600
                interval = self.config.CLM_RETRAIN_INTERVAL_HOURS
                
                if hours_since_retrain >= interval:
                    results[model_type] = True
                    self._retrain_triggered.setdefault(model_type, []).append(
                        f"scheduled ({hours_since_retrain:.1f}h since last)"
                    )
                else:
                    results[model_type] = False
        
        return results
    
    async def trigger_retraining(
        self,
        model_type: ModelType,
        reason: str,
        training_config: Optional[TrainingConfig] = None
    ) -> Dict[str, Any]:
        """
        Manually trigger retraining for a specific model type.
        
        Args:
            model_type: Type of model to retrain
            reason: Reason for retraining
            training_config: Training configuration (optional)
        
        Returns:
            Training result
        """
        self.logger.info(
            f"[CLM] Triggering retraining for {model_type.value}: {reason}"
        )
        
        result = await self.training_daemon.run_training_cycle(
            model_type=model_type,
            trigger=TrainingTrigger.MANUAL,
            reason=reason,
            training_config=training_config
        )
        
        if result["status"] == "success":
            self._last_retrain_time[model_type] = datetime.now(timezone.utc)
            self._retrain_triggered.pop(model_type, None)
        
        return result
    
    async def run_full_cycle(self) -> Dict[str, Any]:
        """
        Run full retraining cycle for all models that need it.
        
        Returns:
            Summary of retraining results
        """
        self.logger.info("[CLM] Starting full retraining cycle")
        
        needs_retrain = await self.check_if_retrain_needed()
        results = []
        
        for model_type, should_retrain in needs_retrain.items():
            if not should_retrain:
                continue
            
            reasons = self._retrain_triggered.get(model_type, ["unknown"])
            reason = ", ".join(reasons)
            
            result = await self.training_daemon.run_training_cycle(
                model_type=model_type,
                trigger=TrainingTrigger.SCHEDULED,
                reason=reason
            )
            
            results.append({
                "model_type": model_type.value,
                "status": result["status"],
                "job_id": result.get("job_id"),
                "model_version": result.get("model_version"),
                "duration_seconds": result.get("duration_seconds")
            })
            
            if result["status"] == "success":
                self._last_retrain_time[model_type] = datetime.now(timezone.utc)
                self._retrain_triggered.pop(model_type, None)
        
        self.logger.info(
            f"[CLM] Full cycle completed: {len(results)} models retrained"
        )
        
        return {
            "cycle_completed_at": datetime.now(timezone.utc).isoformat(),
            "models_retrained": len(results),
            "results": results
        }
    
    def get_last_retrain_times(self) -> Dict[str, str]:
        """Get last retrain time for each model type"""
        return {
            model_type.value: last_time.isoformat()
            for model_type, last_time in self._last_retrain_time.items()
        }
