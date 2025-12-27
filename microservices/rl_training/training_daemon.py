"""
RL Training Daemon

Core training orchestrator for RL models and supervised ML models.
Handles training cycles, model evaluation, and registration.
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from pathlib import Path
import json

from microservices.rl_training.models import (
    ModelType,
    ModelStatus,
    TrainingTrigger,
    ModelTrainingStartedEvent,
    ModelTrainingCompletedEvent,
    TrainingConfig,
    EvaluationResult,
)


logger = logging.getLogger(__name__)


class RLTrainingDaemon:
    """
    Training daemon for RL and ML models.
    
    Responsibilities:
    - Fetch training data from data source
    - Train RL (PPO) and supervised models (XGB, LGBM, etc.)
    - Evaluate trained models
    - Register new model versions in registry
    - Publish training events
    """
    
    def __init__(
        self,
        policy_store,
        data_source,
        model_registry,
        event_bus,
        config,
        logger_instance=None
    ):
        """
        Initialize training daemon.
        
        Args:
            policy_store: PolicyStore for readonly access to trading policies
            data_source: Data source for fetching training data
            model_registry: Model registry for version management
            event_bus: EventBus for publishing events
            config: Service configuration
            logger_instance: Logger instance (optional)
        """
        self.policy_store = policy_store
        self.data_source = data_source
        self.model_registry = model_registry
        self.event_bus = event_bus
        self.config = config
        self.logger = logger_instance or logger
        
        self._running = False
        self._current_job_id: Optional[str] = None
        self._training_history: List[Dict[str, Any]] = []
    
    async def run_training_cycle(
        self,
        model_type: ModelType,
        trigger: TrainingTrigger,
        reason: str,
        training_config: Optional[TrainingConfig] = None
    ) -> Dict[str, Any]:
        """
        Execute complete training cycle.
        
        Steps:
        1. Fetch training data
        2. Train model
        3. Evaluate performance
        4. Register new model version
        5. Publish events
        
        Args:
            model_type: Type of model to train
            trigger: What triggered this training
            reason: Human-readable reason
            training_config: Training configuration (optional)
        
        Returns:
            Training result dictionary
        """
        job_id = f"train_{model_type.value}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        self._current_job_id = job_id
        started_at = datetime.now(timezone.utc)
        
        self.logger.info(
            f"[RLTrainingDaemon] Starting training cycle: {job_id}",
            extra={
                "model_type": model_type.value,
                "trigger": trigger.value,
                "reason": reason
            }
        )
        
        # Publish training started event
        await self.event_bus.publish(
            "model.training_started",
            ModelTrainingStartedEvent(
                job_id=job_id,
                model_type=model_type,
                trigger=trigger,
                reason=reason,
                started_at=started_at.isoformat()
            ).model_dump()
        )
        
        try:
            # Step 1: Fetch training data
            self.logger.info(f"[RLTrainingDaemon] Fetching training data for {model_type.value}")
            training_data = await self._fetch_training_data(
                model_type,
                training_config
            )
            
            if training_data["sample_count"] < self.config.MIN_SAMPLES_FOR_RETRAIN:
                raise ValueError(
                    f"Insufficient samples: {training_data['sample_count']} < "
                    f"{self.config.MIN_SAMPLES_FOR_RETRAIN}"
                )
            
            # Step 2: Train model
            self.logger.info(
                f"[RLTrainingDaemon] Training {model_type.value} with "
                f"{training_data['sample_count']} samples"
            )
            trained_model = await self._train_model(
                model_type,
                training_data,
                training_config
            )
            
            # Step 3: Evaluate performance
            self.logger.info(f"[RLTrainingDaemon] Evaluating {model_type.value}")
            evaluation_result = await self._evaluate_model(
                model_type,
                trained_model,
                training_data
            )
            
            # Step 4: Register new model version
            model_version = await self._register_model(
                model_type,
                trained_model,
                evaluation_result,
                job_id
            )
            
            # Calculate duration
            completed_at = datetime.now(timezone.utc)
            duration_seconds = (completed_at - started_at).total_seconds()
            
            # Publish training completed event
            await self.event_bus.publish(
                "model.training_completed",
                ModelTrainingCompletedEvent(
                    job_id=job_id,
                    model_type=model_type,
                    model_version=model_version,
                    status="success",
                    metrics=evaluation_result.validation_metrics,
                    training_duration_seconds=duration_seconds,
                    completed_at=completed_at.isoformat()
                ).model_dump()
            )
            
            self.logger.info(
                f"[RLTrainingDaemon] Training cycle completed: {job_id}",
                extra={
                    "model_version": model_version,
                    "duration_seconds": duration_seconds,
                    "metrics": evaluation_result.validation_metrics
                }
            )
            
            # Store in history
            self._training_history.append({
                "job_id": job_id,
                "model_type": model_type.value,
                "trigger": trigger.value,
                "status": "success",
                "started_at": started_at.isoformat(),
                "completed_at": completed_at.isoformat(),
                "duration_seconds": duration_seconds,
                "model_version": model_version,
                "metrics": evaluation_result.validation_metrics
            })
            
            return {
                "status": "success",
                "job_id": job_id,
                "model_version": model_version,
                "metrics": evaluation_result.validation_metrics,
                "duration_seconds": duration_seconds
            }
        
        except Exception as e:
            self.logger.error(
                f"[RLTrainingDaemon] Training cycle failed: {job_id}",
                extra={"error": str(e)},
                exc_info=True
            )
            
            completed_at = datetime.now(timezone.utc)
            duration_seconds = (completed_at - started_at).total_seconds()
            
            # Publish training failed event
            await self.event_bus.publish(
                "model.training_completed",
                ModelTrainingCompletedEvent(
                    job_id=job_id,
                    model_type=model_type,
                    model_version="failed",
                    status="failed",
                    metrics={},
                    training_duration_seconds=duration_seconds,
                    completed_at=completed_at.isoformat()
                ).model_dump()
            )
            
            # Store in history
            self._training_history.append({
                "job_id": job_id,
                "model_type": model_type.value,
                "trigger": trigger.value,
                "status": "failed",
                "started_at": started_at.isoformat(),
                "completed_at": completed_at.isoformat(),
                "duration_seconds": duration_seconds,
                "error": str(e)
            })
            
            return {
                "status": "error",
                "job_id": job_id,
                "error": str(e),
                "duration_seconds": duration_seconds
            }
        finally:
            self._current_job_id = None
    
    async def _fetch_training_data(
        self,
        model_type: ModelType,
        training_config: Optional[TrainingConfig]
    ) -> Dict[str, Any]:
        """
        Fetch training data from data source.
        
        Returns:
            Dictionary with:
            - features: Feature matrix
            - labels: Target labels
            - sample_count: Number of samples
            - feature_names: List of feature names
        """
        # Use data_source to fetch historical data
        # For now, this is a stub - in production, this would query:
        # - TradeStore for closed trades
        # - Market data for features
        # - RL replay buffer for RL training
        
        lookback_days = (
            training_config.data_lookback_days if training_config
            else 90
        )
        
        # Placeholder: would call self.data_source.fetch_training_data()
        self.logger.info(
            f"[RLTrainingDaemon] Fetching data: lookback={lookback_days}d"
        )
        
        # Call data source to fetch training data
        return await self.data_source.fetch_training_data(
            lookback_days=lookback_days,
            min_samples=self.config.MIN_SAMPLES_FOR_RETRAIN
        )
    
    async def _train_model(
        self,
        model_type: ModelType,
        training_data: Dict[str, Any],
        training_config: Optional[TrainingConfig]
    ) -> Any:
        """
        Train model.
        
        Returns:
            Trained model object
        """
        # Placeholder: would instantiate and train:
        # - PPO agent for RL_PPO
        # - XGBoost for XGBOOST
        # - etc.
        
        self.logger.info(
            f"[RLTrainingDaemon] Training {model_type.value} "
            f"with {training_data['sample_count']} samples"
        )
        
        # Simulate training delay (minimal for tests/fast feedback)
        await asyncio.sleep(0.001)
        
        return {"model_type": model_type.value, "trained": True}
    
    async def _evaluate_model(
        self,
        model_type: ModelType,
        trained_model: Any,
        training_data: Dict[str, Any]
    ) -> EvaluationResult:
        """
        Evaluate trained model.
        
        Returns:
            EvaluationResult with metrics
        """
        # Placeholder: would run validation and test sets
        # Calculate Sharpe, win rate, max drawdown, etc.
        
        self.logger.info(f"[RLTrainingDaemon] Evaluating {model_type.value}")
        
        # Simulate evaluation delay (minimal for tests/fast feedback)
        await asyncio.sleep(0.001)
        
        return EvaluationResult(
            model_version=f"{model_type.value}_v{datetime.now(timezone.utc).strftime('%Y%m%d')}",
            validation_metrics={
                "sharpe_ratio": 1.85,
                "win_rate": 0.58,
                "max_drawdown": 0.08,
                "total_return": 0.25
            },
            test_metrics={
                "sharpe_ratio": 1.72,
                "win_rate": 0.56,
                "max_drawdown": 0.10,
                "total_return": 0.22
            },
            is_better_than_baseline=True
        )
    
    async def _register_model(
        self,
        model_type: ModelType,
        trained_model: Any,
        evaluation_result: EvaluationResult,
        job_id: str
    ) -> str:
        """
        Register new model version in registry.
        
        Returns:
            Model version string
        """
        model_version = evaluation_result.model_version
        
        # Would call self.model_registry.register_version(...)
        self.logger.info(
            f"[RLTrainingDaemon] Registering {model_type.value} as {model_version}"
        )
        
        return model_version
    
    def get_training_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent training history"""
        return self._training_history[-limit:]
    
    def get_current_job_id(self) -> Optional[str]:
        """Get current running job ID"""
        return self._current_job_id
