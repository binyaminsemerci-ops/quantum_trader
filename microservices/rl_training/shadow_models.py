"""
Shadow Model Manager

Handles shadow model registration, evaluation, and promotion logic.
"""
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

from microservices.rl_training.models import (
    ModelType,
    ModelStatus,
    ModelPromotedEvent,
    ShadowModelStatus,
)


logger = logging.getLogger(__name__)


@dataclass
class ShadowModelRecord:
    """Record for a shadow model"""
    model_name: str
    model_type: ModelType
    version: str
    registered_at: datetime
    num_predictions: int = 0
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    champion_metrics: Dict[str, float] = field(default_factory=dict)


class ShadowModelManager:
    """
    Shadow Model Manager.
    
    Responsibilities:
    - Register new models as shadow (0% allocation)
    - Track shadow predictions and performance
    - Compare shadow vs champion performance
    - Decide on promotion/demotion
    """
    
    def __init__(
        self,
        model_registry,
        event_bus,
        config,
        logger_instance=None
    ):
        """
        Initialize shadow model manager.
        
        Args:
            model_registry: Model registry for version management
            event_bus: EventBus for publishing promotion events
            config: Service configuration
            logger_instance: Logger instance (optional)
        """
        self.model_registry = model_registry
        self.event_bus = event_bus
        self.config = config
        self.logger = logger_instance or logger
        
        self._shadow_models: Dict[str, ShadowModelRecord] = {}
        self._champion_model: Optional[str] = None
    
    async def register_shadow_model(
        self,
        model_type: ModelType,
        version: str,
        model_name: Optional[str] = None
    ) -> str:
        """
        Register a new model as shadow.
        
        Args:
            model_type: Type of model
            version: Model version
            model_name: Optional custom name (defaults to {type}_{version})
        
        Returns:
            Shadow model name
        """
        if model_name is None:
            model_name = f"{model_type.value}_{version}"
        
        shadow_record = ShadowModelRecord(
            model_name=model_name,
            model_type=model_type,
            version=version,
            registered_at=datetime.now(timezone.utc)
        )
        
        self._shadow_models[model_name] = shadow_record
        
        self.logger.info(
            f"[ShadowModelManager] Registered shadow model: {model_name}",
            extra={
                "model_type": model_type.value,
                "version": version
            }
        )
        
        return model_name
    
    async def record_shadow_prediction(
        self,
        model_name: str,
        prediction: Any,
        actual_outcome: Optional[float] = None
    ) -> None:
        """
        Record a prediction from shadow model.
        
        Args:
            model_name: Shadow model name
            prediction: Model prediction
            actual_outcome: Actual outcome (optional, for evaluation)
        """
        if model_name not in self._shadow_models:
            self.logger.warning(
                f"[ShadowModelManager] Unknown shadow model: {model_name}"
            )
            return
        
        shadow = self._shadow_models[model_name]
        shadow.num_predictions += 1
        
        # Store prediction for later evaluation
        # In production, this would store to database or cache
        
        if shadow.num_predictions % 100 == 0:
            self.logger.info(
                f"[ShadowModelManager] Shadow model {model_name} "
                f"has {shadow.num_predictions} predictions"
            )
    
    async def evaluate_shadow_model(
        self,
        model_name: str,
        shadow_metrics: Dict[str, float],
        champion_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Evaluate shadow model performance vs champion.
        
        Args:
            model_name: Shadow model name
            shadow_metrics: Shadow model metrics
            champion_metrics: Champion model metrics
        
        Returns:
            Evaluation result with recommendation
        """
        if model_name not in self._shadow_models:
            return {"status": "error", "message": f"Unknown shadow: {model_name}"}
        
        shadow = self._shadow_models[model_name]
        shadow.performance_metrics = shadow_metrics
        shadow.champion_metrics = champion_metrics
        
        # Check if shadow has enough predictions
        if shadow.num_predictions < self.config.SHADOW_MIN_PREDICTIONS:
            return {
                "status": "insufficient_data",
                "message": f"Need {self.config.SHADOW_MIN_PREDICTIONS} predictions, "
                           f"have {shadow.num_predictions}",
                "ready_for_promotion": False
            }
        
        # Calculate performance differences
        sharpe_diff = (
            shadow_metrics.get("sharpe_ratio", 0) -
            champion_metrics.get("sharpe_ratio", 0)
        )
        winrate_diff = (
            shadow_metrics.get("win_rate", 0) -
            champion_metrics.get("win_rate", 0)
        )
        
        # Check promotion criteria
        min_improvement = self.config.MIN_IMPROVEMENT_FOR_PROMOTION
        ready_for_promotion = (
            sharpe_diff > min_improvement and
            winrate_diff > 0
        )
        
        self.logger.info(
            f"[ShadowModelManager] Evaluated {model_name}: "
            f"sharpe_diff={sharpe_diff:.3f}, winrate_diff={winrate_diff:.3f}, "
            f"ready={ready_for_promotion}"
        )
        
        return {
            "status": "evaluated",
            "model_name": model_name,
            "num_predictions": shadow.num_predictions,
            "sharpe_diff": sharpe_diff,
            "winrate_diff": winrate_diff,
            "ready_for_promotion": ready_for_promotion,
            "recommendation": "promote" if ready_for_promotion else "continue_shadow"
        }
    
    async def promote_shadow_to_active(
        self,
        model_name: str,
        reason: str
    ) -> Dict[str, Any]:
        """
        Promote shadow model to active (champion).
        
        Args:
            model_name: Shadow model name
            reason: Promotion reason
        
        Returns:
            Promotion result
        """
        if model_name not in self._shadow_models:
            return {"status": "error", "message": f"Unknown shadow: {model_name}"}
        
        shadow = self._shadow_models[model_name]
        old_champion = self._champion_model
        
        # Promote shadow to champion
        self._champion_model = model_name
        
        # Move old champion to archive
        if old_champion and old_champion in self._shadow_models:
            self.logger.info(
                f"[ShadowModelManager] Archiving old champion: {old_champion}"
            )
            # Would update model_registry status to RETIRED
        
        # Remove from shadow models (now active)
        self._shadow_models.pop(model_name)
        
        # Calculate improvement
        sharpe_diff = (
            shadow.performance_metrics.get("sharpe_ratio", 0) -
            shadow.champion_metrics.get("sharpe_ratio", 0)
        )
        improvement_pct = sharpe_diff * 100
        
        # Publish promotion event
        await self.event_bus.publish(
            "model.promoted",
            ModelPromotedEvent(
                model_type=shadow.model_type,
                old_version=old_champion if old_champion else None,
                new_version=shadow.version,
                promotion_reason=reason,
                improvement_pct=improvement_pct,
                promoted_at=datetime.now(timezone.utc).isoformat()
            ).model_dump()
        )
        
        self.logger.info(
            f"[ShadowModelManager] Promoted {model_name} to active champion",
            extra={
                "old_champion": old_champion,
                "improvement_pct": improvement_pct
            }
        )
        
        return {
            "status": "promoted",
            "model_name": model_name,
            "old_champion": old_champion,
            "improvement_pct": improvement_pct
        }
    
    def get_shadow_models(self) -> List[ShadowModelStatus]:
        """Get all shadow models"""
        results = []
        
        for model_name, shadow in self._shadow_models.items():
            # Calculate performance vs champion
            if shadow.champion_metrics:
                sharpe_diff = (
                    shadow.performance_metrics.get("sharpe_ratio", 0) -
                    shadow.champion_metrics.get("sharpe_ratio", 0)
                )
                winrate_diff = (
                    shadow.performance_metrics.get("win_rate", 0) -
                    shadow.champion_metrics.get("win_rate", 0)
                )
                performance_vs_champion = {
                    "sharpe_diff": sharpe_diff,
                    "winrate_diff": winrate_diff
                }
            else:
                performance_vs_champion = {}
            
            # Check if ready for promotion
            ready = (
                shadow.num_predictions >= self.config.SHADOW_MIN_PREDICTIONS and
                performance_vs_champion.get("sharpe_diff", 0) > self.config.MIN_IMPROVEMENT_FOR_PROMOTION
            )
            
            results.append(
                ShadowModelStatus(
                    model_name=model_name,
                    model_type=shadow.model_type,
                    version=shadow.version,
                    shadow_since=shadow.registered_at.isoformat(),
                    num_predictions=shadow.num_predictions,
                    performance_vs_champion=performance_vs_champion,
                    ready_for_promotion=ready
                )
            )
        
        return results
    
    def get_champion_model(self) -> Optional[str]:
        """Get current champion model name"""
        return self._champion_model
