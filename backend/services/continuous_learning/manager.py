"""
Continuous Learning Manager - orchestrates model retraining and promotion.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from ..eventbus import InMemoryEventBus, ModelPromotedEvent
from .models import (
    ModelVersion,
    RetrainingConfig,
    ModelStage,
)
from .evaluator import ShadowEvaluator

logger = logging.getLogger(__name__)


class ContinuousLearningManager:
    """
    Manages the continuous learning lifecycle for ML models.
    
    - Schedules model retraining
    - Deploys to shadow mode
    - Evaluates performance
    - Promotes models when better
    """
    
    def __init__(
        self,
        eventbus: InMemoryEventBus,
        evaluator: Optional[ShadowEvaluator] = None,
    ):
        self.eventbus = eventbus
        self.evaluator = evaluator or ShadowEvaluator()
        
        self._models: Dict[str, ModelVersion] = {}
        self._configs: Dict[str, RetrainingConfig] = {}
        self._last_retrain: Dict[str, datetime] = {}
        self._running = False
    
    def register_model(
        self,
        model_name: str,
        initial_version: ModelVersion,
        config: RetrainingConfig,
    ):
        """Register a model for continuous learning."""
        self._models[model_name] = initial_version
        self._configs[model_name] = config
        self._last_retrain[model_name] = datetime.now()
        
        logger.info(f"Registered model {model_name} for continuous learning")
    
    def get_model(self, model_name: str) -> Optional[ModelVersion]:
        """Get current model version."""
        return self._models.get(model_name)
    
    def get_live_model(self, model_name: str) -> Optional[ModelVersion]:
        """Get the live model version."""
        model = self._models.get(model_name)
        if model and model.stage == ModelStage.LIVE:
            return model
        return None
    
    async def retrain_model(
        self,
        model_name: str,
        training_data: Optional[list] = None,
    ) -> ModelVersion:
        """
        Retrain a model with new data.
        
        This is a simplified implementation - in production,
        this would call actual ML training pipeline.
        """
        config = self._configs.get(model_name)
        if not config:
            raise ValueError(f"Model {model_name} not registered")
        
        current_model = self._models.get(model_name)
        if not current_model:
            raise ValueError(f"Model {model_name} not found")
        
        # Generate new version
        new_version = f"{float(current_model.version) + 0.1:.1f}"
        
        logger.info(f"Retraining {model_name} to version {new_version}...")
        
        # Simulate training (in production, this would train actual model)
        await asyncio.sleep(0.1)
        
        # Create new model version in SHADOW stage
        new_model = ModelVersion(
            model_name=model_name,
            version=new_version,
            stage=ModelStage.SHADOW,
            model_type=current_model.model_type,
            trained_on_samples=len(training_data) if training_data else 1000,
            training_time_seconds=10.0,
            hyperparameters=config.hyperparameters,
        )
        
        # Update last retrain time
        self._last_retrain[model_name] = datetime.now()
        
        logger.info(f"Trained {model_name} v{new_version} - deploying to SHADOW")
        
        return new_model
    
    async def evaluate_and_promote(
        self,
        model_name: str,
        shadow_model: ModelVersion,
    ) -> bool:
        """
        Evaluate shadow model and promote if better.
        
        Returns True if promoted.
        """
        live_model = self.get_live_model(model_name)
        if not live_model:
            # No live model, promote shadow directly
            logger.info(f"No live model for {model_name}, promoting shadow v{shadow_model.version}")
            await self.promote_model(shadow_model)
            return True
        
        config = self._configs[model_name]
        
        # Evaluate shadow vs live
        evaluation = await self.evaluator.evaluate(
            shadow_model=shadow_model,
            live_model=live_model,
            evaluation_hours=config.shadow_evaluation_hours,
        )
        
        logger.info(
            f"Evaluation for {model_name}: "
            f"Accuracy improvement: {evaluation.accuracy_improvement:.3f}, "
            f"Should promote: {evaluation.should_promote}"
        )
        
        if evaluation.should_promote:
            await self.promote_model(shadow_model)
            
            # Retire old model
            live_model.stage = ModelStage.RETIRED
            live_model.retired_at = datetime.now()
            
            return True
        
        return False
    
    async def promote_model(self, model: ModelVersion):
        """[CRITICAL FIX #1] Promote a model to LIVE stage with atomic lock."""
        # Acquire promotion lock before updating any state
        required_handlers = ["ensemble_manager", "meta_strategy", "sesa"]
        lock_acquired = await self.eventbus.acquire_promotion_lock(required_handlers)
        
        if not lock_acquired:
            logger.error("Failed to acquire promotion lock - another promotion in progress")
            return
        
        try:
            # Update model stage
            model.stage = ModelStage.LIVE
            model.promoted_to_live_at = datetime.now()
            
            # Update stored model
            self._models[model.model_name] = model
            
            # Publish ModelPromotedEvent with priority routing
            event = ModelPromotedEvent.create(
                model_name=model.model_name,
                old_version="0.0",  # Simplified
                new_version=model.version,
                metrics={
                    "accuracy": model.accuracy,
                    "precision": model.precision,
                    "recall": model.recall,
                    "f1_score": model.f1_score,
                },
            )
            
            await self.eventbus.publish(event)
            logger.info(f"[PROMOTION-LOCK] Published model.promoted event for {model.model_name} v{model.version}")
            
            # Wait for all handlers to ACK (30s timeout)
            all_acked = await self.eventbus.wait_for_promotion_acks(timeout=30.0)
            
            if not all_acked:
                logger.error(f"Not all handlers ACKed promotion - system may be in mixed state")
            else:
                logger.info(f"Promoted {model.model_name} v{model.version} to LIVE (all handlers synchronized)")
        
        finally:
            # Always release lock
            await self.eventbus.release_promotion_lock()
    
    def should_retrain(self, model_name: str) -> bool:
        """Check if model should be retrained."""
        config = self._configs.get(model_name)
        if not config:
            return False
        
        last_retrain = self._last_retrain.get(model_name)
        if not last_retrain:
            return True
        
        hours_since_retrain = (datetime.now() - last_retrain).total_seconds() / 3600
        
        return hours_since_retrain >= config.retrain_interval_hours
    
    async def run_forever(self, check_interval_seconds: int = 3600):
        """Continuously check for retraining needs."""
        self._running = True
        logger.info("ContinuousLearningManager started")
        
        while self._running:
            try:
                # Check each registered model
                for model_name in list(self._models.keys()):
                    if self.should_retrain(model_name):
                        logger.info(f"Retraining {model_name}...")
                        
                        # Retrain model
                        new_model = await self.retrain_model(model_name)
                        
                        # Start shadow evaluation
                        # In production, this would run in parallel
                        # For now, we'll evaluate immediately
                        await asyncio.sleep(1)  # Simulate shadow period
                        
                        # Try to promote
                        promoted = await self.evaluate_and_promote(model_name, new_model)
                        
                        if promoted:
                            logger.info(f"Successfully promoted {model_name} v{new_model.version}")
                        else:
                            logger.info(f"Shadow model {model_name} v{new_model.version} not promoted")
                
                # Wait for next check
                await asyncio.sleep(check_interval_seconds)
                
            except asyncio.CancelledError:
                logger.info("ContinuousLearningManager cancelled")
                break
            except Exception as e:
                logger.error(f"Error in CLM loop: {e}")
                await asyncio.sleep(check_interval_seconds)
        
        logger.info("ContinuousLearningManager stopped")
    
    def stop(self):
        """Stop the CLM loop."""
        self._running = False
