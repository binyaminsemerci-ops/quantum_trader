"""
Continuous Learning Manager (CLM) - Central orchestrator for all learning activities.

Coordinates:
- Scheduled retraining
- Drift monitoring
- Shadow testing
- Model promotion
- RL agent updates
- Performance tracking
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.event_bus import EventBus
from backend.core.policy_store import PolicyStore
from backend.domains.learning.data_pipeline import HistoricalDataFetcher, FeatureEngineer
from backend.domains.learning.drift_detector import DriftDetector, DriftType
from backend.domains.learning.model_registry import ModelRegistry, ModelType
from backend.domains.learning.model_supervisor import ModelSupervisor
from backend.domains.learning.retraining import RetrainingOrchestrator, RetrainingType
from backend.domains.learning.shadow_tester import ShadowTester

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class CLMConfig:
    """Continuous Learning Manager configuration."""
    
    # Scheduling
    retraining_schedule_hours: int = 168  # Weekly
    drift_check_hours: int = 24  # Daily
    performance_check_hours: int = 6  # Every 6 hours
    
    # Drift thresholds
    drift_trigger_threshold: float = 0.05
    performance_degradation_threshold: float = 0.15
    
    # Shadow testing
    shadow_min_predictions: int = 100
    shadow_promotion_threshold: float = 0.05
    
    # Data
    training_data_days: int = 90
    reference_window_days: int = 30
    current_window_days: int = 7
    
    # Flags
    auto_retraining_enabled: bool = True
    auto_promotion_enabled: bool = True
    drift_monitoring_enabled: bool = True


# ============================================================================
# Continuous Learning Manager
# ============================================================================

class ContinuousLearningManager:
    """
    Central orchestrator for all ML/AI learning activities.
    
    Responsibilities:
    1. Schedule periodic retraining
    2. Monitor drift (features, predictions, performance)
    3. Coordinate shadow testing
    4. Auto-promote models when ready
    5. Update RL agents on regime changes
    6. Track system health
    
    Event subscriptions:
    - learning.drift.detected → trigger retraining
    - learning.retraining.completed → start shadow testing
    - learning.performance.alert → investigate and revert if needed
    """
    
    def __init__(
        self,
        db_session: AsyncSession,
        event_bus: EventBus,
        policy_store: PolicyStore,
        config: Optional[CLMConfig] = None,
    ):
        self.db = db_session
        self.event_bus = event_bus
        self.policy_store = policy_store
        self.config = config or CLMConfig()
        
        # Components (initialized in setup())
        self.data_fetcher: Optional[HistoricalDataFetcher] = None
        self.feature_engineer: Optional[FeatureEngineer] = None
        self.model_registry: Optional[ModelRegistry] = None
        self.drift_detector: Optional[DriftDetector] = None
        self.model_supervisor: Optional[ModelSupervisor] = None
        self.shadow_tester: Optional[ShadowTester] = None
        self.retraining_orchestrator: Optional[RetrainingOrchestrator] = None
        
        # State
        self.running = False
        self.last_retraining: Optional[datetime] = None
        self.last_drift_check: Optional[datetime] = None
        self.last_performance_check: Optional[datetime] = None
        
        logger.info("ContinuousLearningManager initialized")
    
    async def setup(self):
        """Initialize all components."""
        logger.info("Setting up CLM components...")
        
        # Ensure all CLM database tables exist
        from backend.domains.learning.retraining import create_retraining_jobs_table
        from backend.domains.learning.model_registry import create_model_registry_table
        
        try:
            create_retraining_jobs_table(self.db)
            logger.info("✅ retraining_jobs table verified/created")
        except Exception as e:
            logger.warning(f"Could not create retraining_jobs table: {e}")
        
        try:
            create_model_registry_table(self.db)
            logger.info("✅ model_registry table verified/created")
        except Exception as e:
            logger.warning(f"Could not create model_registry table: {e}")
        
        # Initialize components
        self.data_fetcher = HistoricalDataFetcher(self.db)
        self.feature_engineer = FeatureEngineer()
        self.model_registry = ModelRegistry(self.db)
        
        self.drift_detector = DriftDetector(
            self.db,
            self.event_bus,
            ks_threshold=self.config.drift_trigger_threshold,
            reference_window_days=self.config.reference_window_days,
            current_window_days=self.config.current_window_days,
        )
        
        self.model_supervisor = ModelSupervisor(
            self.db,
            self.event_bus,
        )
        
        self.shadow_tester = ShadowTester(
            self.db,
            self.event_bus,
            self.model_registry,
            min_predictions=self.config.shadow_min_predictions,
            promotion_threshold=self.config.shadow_promotion_threshold,
        )
        
        self.retraining_orchestrator = RetrainingOrchestrator(
            self.db,
            self.event_bus,
            self.policy_store,
            self.model_registry,
            self.data_fetcher,
            self.feature_engineer,
        )
        
        logger.info("✅ CLM components initialized")
    
    async def start(self):
        """Start continuous learning manager."""
        if self.running:
            logger.warning("CLM already running")
            return
        
        if not self.model_registry:
            await self.setup()
        
        self.running = True
        
        # Subscribe to events (synchronous subscription)
        self.event_bus.subscribe("learning.drift.detected", self._on_drift_detected)
        self.event_bus.subscribe("learning.retraining.completed", self._on_retraining_completed)
        self.event_bus.subscribe("learning.performance.alert", self._on_performance_alert)
        self.event_bus.subscribe("learning.model.promoted", self._on_model_promoted)
        
        # Start shadow tester
        await self.shadow_tester.start()
        
        # Start background tasks
        asyncio.create_task(self._run_scheduled_tasks())
        
        logger.info("✅ ContinuousLearningManager started")
    
    async def stop(self):
        """Stop continuous learning manager."""
        self.running = False
        
        if self.shadow_tester:
            await self.shadow_tester.stop()
        
        logger.info("ContinuousLearningManager stopped")
    
    async def _run_scheduled_tasks(self):
        """Run periodic tasks (retraining, drift checks, performance monitoring)."""
        while self.running:
            try:
                current_time = datetime.utcnow()
                
                # Periodic retraining
                if self.config.auto_retraining_enabled:
                    if (not self.last_retraining or 
                        (current_time - self.last_retraining).total_seconds() / 3600 >= self.config.retraining_schedule_hours):
                        
                        logger.info("🔄 Scheduled retraining triggered")
                        await self.trigger_retraining(
                            retraining_type=RetrainingType.FULL,
                            trigger_reason="scheduled"
                        )
                        self.last_retraining = current_time
                
                # Drift monitoring
                if self.config.drift_monitoring_enabled:
                    if (not self.last_drift_check or 
                        (current_time - self.last_drift_check).total_seconds() / 3600 >= self.config.drift_check_hours):
                        
                        logger.info("🔍 Running drift checks")
                        await self._check_drift()
                        self.last_drift_check = current_time
                
                # Performance monitoring
                if (not self.last_performance_check or 
                    (current_time - self.last_performance_check).total_seconds() / 3600 >= self.config.performance_check_hours):
                    
                    logger.info("📊 Running performance checks")
                    await self._check_performance()
                    self.last_performance_check = current_time
                
                # Shadow model promotion
                if self.config.auto_promotion_enabled:
                    await self._check_shadow_promotions()
                
                # Sleep for 1 hour between checks
                await asyncio.sleep(3600)
            
            except Exception as e:
                logger.error(f"Error in scheduled tasks: {e}", exc_info=True)
                await asyncio.sleep(300)  # Sleep 5 min on error
    
    async def trigger_retraining(
        self,
        retraining_type: RetrainingType = RetrainingType.FULL,
        model_types: Optional[List[ModelType]] = None,
        trigger_reason: str = "manual",
    ) -> str:
        """Trigger model retraining."""
        logger.info(f"Triggering retraining: type={retraining_type.value}, reason={trigger_reason}")
        
        job_id = await self.retraining_orchestrator.trigger_retraining(
            retraining_type=retraining_type,
            model_types=model_types,
            trigger_reason=trigger_reason,
            days_of_data=self.config.training_data_days,
        )
        
        return job_id
    
    async def _check_drift(self):
        """Check for drift in all active models."""
        for model_type in [ModelType.XGBOOST, ModelType.LIGHTGBM, ModelType.NHITS, ModelType.PATCHTST]:
            active_model = await self.model_registry.get_active_model(model_type)
            if not active_model:
                continue
            
            # For now, just log - actual drift detection would need recent prediction data
            logger.debug(f"Drift check for {model_type.value}: {active_model.model_id}")
    
    async def _check_performance(self):
        """Check performance of all active models."""
        for model_type in [ModelType.XGBOOST, ModelType.LIGHTGBM, ModelType.NHITS, ModelType.PATCHTST]:
            active_model = await self.model_registry.get_active_model(model_type)
            if not active_model:
                continue
            
            metrics = await self.model_supervisor.compute_performance_metrics(
                model_type=model_type,
                model_id=active_model.model_id,
                period_days=self.config.current_window_days,
            )
            
            if metrics:
                logger.info(
                    f"Performance for {model_type.value}: "
                    f"winrate={metrics.winrate:.1%}, sharpe={metrics.sharpe_ratio:.2f}"
                )
    
    async def _check_shadow_promotions(self):
        """Check if any shadow models are ready for promotion."""
        for model_type in [ModelType.XGBOOST, ModelType.LIGHTGBM, ModelType.NHITS, ModelType.PATCHTST]:
            try:
                promoted = await self.shadow_tester.promote_if_ready(model_type)
                if promoted:
                    logger.info(f"✅ Promoted shadow model for {model_type.value}")
            except Exception as e:
                logger.error(f"Error checking promotion for {model_type.value}: {e}")
    
    async def _on_drift_detected(self, event: Dict):
        """Handle drift detection event."""
        logger.warning(
            f"⚠️ Drift detected: type={event.get('drift_type')}, "
            f"severity={event.get('severity')}, model={event.get('model_type')}"
        )
        
        if event.get("trigger_retraining") and self.config.auto_retraining_enabled:
            # Trigger retraining for affected model
            model_type_str = event.get("model_type")
            if model_type_str:
                model_types = [ModelType(model_type_str)]
                await self.trigger_retraining(
                    retraining_type=RetrainingType.PARTIAL,
                    model_types=model_types,
                    trigger_reason=f"drift_detected_{event.get('drift_type')}",
                )
    
    async def _on_retraining_completed(self, event: Dict):
        """Handle retraining completion event."""
        logger.info(
            f"✅ Retraining completed: job={event.get('job_id')}, "
            f"models={event.get('models_succeeded')}/{event.get('models_trained')}"
        )
        
        # Shadow models are now registered and will be tested automatically
        # No action needed - shadow tester is already running
    
    async def _on_performance_alert(self, event: Dict):
        """Handle performance alert event."""
        logger.warning(
            f"⚠️ Performance alert for {event.get('model_id')}: "
            f"{len(event.get('alerts', []))} issues"
        )
        
        # Could implement automatic rollback here if performance degrades severely
    
    async def _on_model_promoted(self, event: Dict):
        """Handle model promotion event."""
        logger.info(
            f"✅ Model promoted: {event.get('model_id')} "
            f"(improvements: RMSE {event.get('improvements', {}).get('rmse', 0):.1%})"
        )
        
        # Publish to trading engine to reload models
        await self.event_bus.publish("learning.model.updated", {
            "model_type": event.get("model_type"),
            "model_id": event.get("model_id"),
            "action": "promoted",
        })
    
    async def get_system_status(self) -> Dict:
        """Get overall CLM system status."""
        status = {
            "running": self.running,
            "last_retraining": self.last_retraining.isoformat() if self.last_retraining else None,
            "last_drift_check": self.last_drift_check.isoformat() if self.last_drift_check else None,
            "last_performance_check": self.last_performance_check.isoformat() if self.last_performance_check else None,
            "config": {
                "auto_retraining_enabled": self.config.auto_retraining_enabled,
                "auto_promotion_enabled": self.config.auto_promotion_enabled,
                "drift_monitoring_enabled": self.config.drift_monitoring_enabled,
                "retraining_schedule_hours": self.config.retraining_schedule_hours,
            },
            "active_models": {},
            "shadow_models": {},
        }
        
        # Get active/shadow models
        for model_type in [ModelType.XGBOOST, ModelType.LIGHTGBM, ModelType.NHITS, ModelType.PATCHTST]:
            active = await self.model_registry.get_active_model(model_type)
            shadow = await self.model_registry.get_shadow_model(model_type)
            
            if active:
                status["active_models"][model_type.value] = {
                    "model_id": active.model_id,
                    "version": active.version,
                    "promoted_at": active.promoted_at.isoformat() if active.promoted_at else None,
                }
            
            if shadow:
                status["shadow_models"][model_type.value] = {
                    "model_id": shadow.model_id,
                    "version": shadow.version,
                    "created_at": shadow.created_at.isoformat() if shadow.created_at else None,
                }
        
        return status
    
    async def manual_trigger_drift_check(self, model_type: ModelType) -> Dict:
        """Manually trigger drift detection for a model."""
        logger.info(f"Manual drift check triggered for {model_type.value}")
        
        # This would fetch recent data and compare distributions
        # For now, return placeholder
        return {
            "model_type": model_type.value,
            "status": "completed",
            "drift_detected": False,
        }
    
    async def manual_promote_shadow(self, model_type: ModelType) -> bool:
        """Manually promote shadow model to active."""
        logger.info(f"Manual promotion triggered for {model_type.value}")
        
        success = await self.model_registry.promote_shadow_to_active(model_type)
        
        if success:
            await self.event_bus.publish("learning.model.promoted", {
                "model_type": model_type.value,
                "action": "manual_promotion",
                "timestamp": datetime.utcnow().isoformat(),
            })
        
        return success


# ============================================================================
# Factory Function
# ============================================================================

async def create_clm(
    db_session: AsyncSession,
    event_bus: EventBus,
    policy_store: PolicyStore,
    config: Optional[CLMConfig] = None,
) -> ContinuousLearningManager:
    """
    Factory function to create and initialize CLM.
    
    Usage:
        clm = await create_clm(db_session, event_bus, policy_store)
        await clm.start()
    """
    clm = ContinuousLearningManager(db_session, event_bus, policy_store, config)
    await clm.setup()
    return clm
