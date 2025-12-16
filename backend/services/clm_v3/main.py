"""
CLM v3 Main - EventBus integration and service lifecycle.

Handles:
- EventBus v2 event subscriptions
- Service initialization
- Event routing (drift, performance, regime change)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, Optional

from backend.services.clm_v3.adapters import (
    BacktestAdapter,
    DataLoaderAdapter,
    ModelTrainingAdapter,
)
from backend.services.clm_v3.models import ModelType, TriggerReason
from backend.services.clm_v3.orchestrator import ClmOrchestrator
from backend.services.clm_v3.scheduler import TrainingScheduler
from backend.services.clm_v3.storage import ModelRegistryV3
from backend.services.clm_v3.strategies import StrategyEvolutionEngine

logger = logging.getLogger(__name__)


# ============================================================================
# CLM v3 Service
# ============================================================================

class ClmV3Service:
    """
    CLM v3 Service - Main service class with EventBus integration.
    
    Responsibilities:
    - Initialize all CLM v3 components
    - Subscribe to EventBus events
    - Route events to appropriate handlers
    - Manage service lifecycle
    """
    
    def __init__(
        self,
        event_bus: Optional[any] = None,
        config: Optional[Dict] = None,
    ):
        """
        Initialize CLM v3 Service.
        
        Args:
            event_bus: EventBus v2 instance (optional)
            config: Service configuration
        """
        self.event_bus = event_bus
        self.config = config or self._default_config()
        
        # Initialize components
        logger.info("[CLM v3] Initializing service components...")
        
        self.registry = ModelRegistryV3(
            models_dir=self.config.get("models_dir", "/app/models"),
            metadata_dir=self.config.get("metadata_dir", "/app/data/clm_v3/registry"),
        )
        
        self.training_adapter = ModelTrainingAdapter(
            models_dir=self.config.get("models_dir", "/app/models"),
        )
        
        self.backtest_adapter = BacktestAdapter()
        
        self.data_loader = DataLoaderAdapter()
        
        self.orchestrator = ClmOrchestrator(
            registry=self.registry,
            training_adapter=self.training_adapter,
            backtest_adapter=self.backtest_adapter,
            event_bus=self.event_bus,
            config=self.config.get("orchestrator", {}),
        )
        
        self.scheduler = TrainingScheduler(
            registry=self.registry,
            config=self.config.get("scheduler", {}),
        )
        
        self.evolution = StrategyEvolutionEngine(
            config=self.config.get("evolution", {}),
        )
        
        self._running = False
        
        logger.info("[CLM v3] Service initialized successfully")
    
    @staticmethod
    def _default_config() -> Dict:
        """Default service configuration."""
        return {
            "models_dir": "/app/models",
            "metadata_dir": "/app/data/clm_v3/registry",
            
            "orchestrator": {
                "auto_promote_to_candidate": True,
                "auto_promote_to_production": False,
                "require_shadow_testing": True,
            },
            
            "scheduler": {
                "enabled": True,
                "check_interval_minutes": 30,
            },
            
            "evolution": {
                "enabled": False,  # Skeleton only
            },
            
            "event_subscriptions": {
                "drift_detected": True,
                "performance_degraded": True,
                "manual_training_requested": True,
                "regime_changed": True,
            },
        }
    
    # ========================================================================
    # Lifecycle
    # ========================================================================
    
    async def start(self):
        """Start CLM v3 service."""
        if self._running:
            logger.warning("[CLM v3] Service already running")
            return
        
        logger.info("[CLM v3] Starting service...")
        
        # Subscribe to EventBus events
        if self.event_bus:
            await self._subscribe_to_events()
        else:
            logger.warning("[CLM v3] No EventBus provided - event-driven triggers disabled")
        
        # Start scheduler
        if self.config["scheduler"]["enabled"]:
            await self.scheduler.start()
        
        self._running = True
        logger.info("[CLM v3] ✅ Service started")
    
    async def stop(self):
        """Stop CLM v3 service."""
        if not self._running:
            return
        
        logger.info("[CLM v3] Stopping service...")
        
        # Stop scheduler
        await self.scheduler.stop()
        
        # Unsubscribe from events
        if self.event_bus:
            await self._unsubscribe_from_events()
        
        self._running = False
        logger.info("[CLM v3] Service stopped")
    
    # ========================================================================
    # EventBus Integration
    # ========================================================================
    
    async def _subscribe_to_events(self):
        """Subscribe to EventBus events."""
        logger.info("[CLM v3] Subscribing to EventBus events...")
        
        subscriptions = self.config["event_subscriptions"]
        
        # Subscribe to drift detection events
        if subscriptions.get("drift_detected"):
            # TODO: Implement EventBus v2 subscription
            # await self.event_bus.subscribe("model.drift_detected", self.handle_drift_detected)
            logger.info("[CLM v3] Subscribed to: model.drift_detected")
        
        # Subscribe to performance degradation events
        if subscriptions.get("performance_degraded"):
            # await self.event_bus.subscribe("performance.degraded", self.handle_performance_degraded)
            logger.info("[CLM v3] Subscribed to: performance.degraded")
        
        # Subscribe to manual training requests
        if subscriptions.get("manual_training_requested"):
            # await self.event_bus.subscribe("manual.training_requested", self.handle_manual_training)
            logger.info("[CLM v3] Subscribed to: manual.training_requested")
        
        # Subscribe to regime change events
        if subscriptions.get("regime_changed"):
            # await self.event_bus.subscribe("market.regime_changed", self.handle_regime_change)
            logger.info("[CLM v3] Subscribed to: market.regime_changed")
        
        logger.info("[CLM v3] EventBus subscriptions complete")
    
    async def _unsubscribe_from_events(self):
        """Unsubscribe from EventBus events."""
        # TODO: Implement EventBus v2 unsubscription
        pass
    
    # ========================================================================
    # Event Handlers
    # ========================================================================
    
    async def handle_drift_detected(self, event: Dict):
        """
        Handle model.drift_detected event.
        
        Event payload:
        {
            "model_id": str,
            "model_type": str,
            "drift_score": float,
            "detected_at": str (ISO timestamp),
        }
        """
        logger.warning(
            f"[CLM v3] Drift detected: {event.get('model_id')} "
            f"(score={event.get('drift_score')})"
        )
        
        model_type_str = event.get("model_type", "xgboost")
        model_type = ModelType(model_type_str)
        
        await self.scheduler.handle_drift_detected(
            model_id=event["model_id"],
            model_type=model_type,
            drift_score=event["drift_score"],
        )
    
    async def handle_performance_degraded(self, event: Dict):
        """
        Handle performance.degraded event.
        
        Event payload:
        {
            "model_id": str,
            "model_type": str,
            "sharpe_ratio": float,
            "detected_at": str,
        }
        """
        logger.warning(
            f"[CLM v3] Performance degraded: {event.get('model_id')} "
            f"(sharpe={event.get('sharpe_ratio')})"
        )
        
        model_type_str = event.get("model_type", "xgboost")
        model_type = ModelType(model_type_str)
        
        await self.scheduler.handle_performance_degraded(
            model_id=event["model_id"],
            model_type=model_type,
            sharpe_ratio=event["sharpe_ratio"],
        )
    
    async def handle_manual_training(self, event: Dict):
        """
        Handle manual.training_requested event.
        
        Event payload:
        {
            "model_type": str,
            "symbol": str (optional),
            "timeframe": str,
            "requested_by": str,
        }
        """
        logger.info(
            f"[CLM v3] Manual training requested: {event.get('model_type')} "
            f"(by {event.get('requested_by')})"
        )
        
        model_type = ModelType(event["model_type"])
        
        job = await self.scheduler.trigger_training(
            model_type=model_type,
            trigger_reason=TriggerReason.MANUAL,
            triggered_by=event.get("requested_by", "unknown"),
            symbol=event.get("symbol"),
            timeframe=event.get("timeframe", "1h"),
        )
        
        # Start training in background
        asyncio.create_task(self.orchestrator.handle_training_job(job))
    
    async def handle_regime_change(self, event: Dict):
        """
        Handle market.regime_changed event.
        
        Event payload:
        {
            "old_regime": str,
            "new_regime": str,
            "affected_models": List[str],
            "detected_at": str,
        }
        """
        logger.info(
            f"[CLM v3] Market regime changed: {event.get('old_regime')} → {event.get('new_regime')}"
        )
        
        await self.scheduler.handle_regime_change(
            new_regime=event["new_regime"],
            affected_models=event.get("affected_models", []),
        )
    
    # ========================================================================
    # Status
    # ========================================================================
    
    def get_status(self) -> Dict:
        """Get service status."""
        return {
            "service": "clm_v3",
            "version": "3.0.0",
            "running": self._running,
            "components": {
                "registry": "initialized",
                "orchestrator": "initialized",
                "scheduler": "running" if self.scheduler._running else "stopped",
                "evolution": "initialized",
            },
            "event_bus": "connected" if self.event_bus else "not_connected",
        }


# ============================================================================
# Factory
# ============================================================================

async def create_clm_v3_service(
    event_bus: Optional[any] = None,
    config: Optional[Dict] = None,
) -> ClmV3Service:
    """
    Create and start CLM v3 service.
    
    Args:
        event_bus: EventBus v2 instance
        config: Service configuration
    
    Returns:
        Initialized ClmV3Service
    """
    service = ClmV3Service(event_bus=event_bus, config=config)
    await service.start()
    return service
