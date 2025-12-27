"""
Event Handlers

Handles incoming events from EventBus.
"""
import logging
from typing import Dict, Any

from microservices.rl_training.models import (
    ModelType,
    TrainingTrigger,
    PerformanceMetricsUpdatedEvent,
    DataDriftSignalEvent,
    ManualRetrainRequestEvent,
)


logger = logging.getLogger(__name__)


class EventHandlers:
    """
    Event handlers for RL Training Service.
    
    Handles:
    - performance.metrics_updated
    - data.drift_signal
    - manual.retrain_request
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
        Initialize event handlers.
        
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
    
    async def handle_performance_metrics_updated(
        self,
        event_data: Dict[str, Any]
    ) -> None:
        """
        Handle performance.metrics_updated event.
        
        Checks if performance has degraded and triggers retraining if needed.
        """
        try:
            event = PerformanceMetricsUpdatedEvent(**event_data)
            
            self.logger.info(
                f"[EventHandlers] Performance metrics updated: {event.model_name}",
                extra={
                    "win_rate": event.win_rate,
                    "sharpe_ratio": event.sharpe_ratio,
                    "max_drawdown_pct": event.max_drawdown_pct
                }
            )
            
            # Check if performance has degraded
            # (Would compare against baseline metrics from model registry)
            baseline_metrics = {
                "sharpe_ratio": 1.5,  # Placeholder
                "win_rate": 0.55
            }
            
            current_metrics = {
                "sharpe_ratio": event.sharpe_ratio,
                "win_rate": event.win_rate
            }
            
            degradation_result = await self.drift_detector.check_performance_degradation(
                model_name=event.model_name,
                current_metrics=current_metrics,
                baseline_metrics=baseline_metrics
            )
            
            if degradation_result["degraded"]:
                self.logger.warning(
                    f"[EventHandlers] Performance degraded for {event.model_name}, "
                    "triggering retraining"
                )
                
                # Trigger retraining
                # (Would map model_name to ModelType)
                model_type = ModelType.XGBOOST  # Placeholder
                
                await self.training_daemon.run_training_cycle(
                    model_type=model_type,
                    trigger=TrainingTrigger.PERFORMANCE_DECAY,
                    reason=f"Performance degraded: sharpe={degradation_result['sharpe_change']:.3f}"
                )
        
        except Exception as e:
            self.logger.error(
                f"[EventHandlers] Error handling performance_metrics_updated: {e}",
                exc_info=True
            )
    
    async def handle_data_drift_signal(
        self,
        event_data: Dict[str, Any]
    ) -> None:
        """
        Handle data.drift_signal event.
        
        Logs drift and may trigger retraining.
        """
        try:
            event = DataDriftSignalEvent(**event_data)
            
            self.logger.info(
                f"[EventHandlers] Data drift signal received",
                extra={
                    "drift_type": event.drift_type,
                    "severity": event.severity.value,
                    "psi_score": event.psi_score
                }
            )
            
            # If severe drift, trigger retraining
            if event.severity in ["severe", "critical"]:
                self.logger.warning(
                    f"[EventHandlers] Severe drift detected, triggering retraining"
                )
                
                # Trigger retraining for affected models
                # (Would determine affected models from event)
                model_type = ModelType.XGBOOST  # Placeholder
                
                await self.training_daemon.run_training_cycle(
                    model_type=model_type,
                    trigger=TrainingTrigger.DRIFT_DETECTED,
                    reason=f"Drift detected: {event.drift_type}, PSI={event.psi_score:.3f}"
                )
        
        except Exception as e:
            self.logger.error(
                f"[EventHandlers] Error handling data_drift_signal: {e}",
                exc_info=True
            )
    
    async def handle_manual_retrain_request(
        self,
        event_data: Dict[str, Any]
    ) -> None:
        """
        Handle manual.retrain_request event.
        
        Triggers immediate retraining.
        """
        try:
            event = ManualRetrainRequestEvent(**event_data)
            
            self.logger.info(
                f"[EventHandlers] Manual retrain request received",
                extra={
                    "model_type": event.model_type.value,
                    "reason": event.reason,
                    "requested_by": event.requested_by,
                    "priority": event.priority
                }
            )
            
            # Trigger retraining immediately
            await self.training_daemon.run_training_cycle(
                model_type=event.model_type,
                trigger=TrainingTrigger.MANUAL,
                reason=f"Manual request by {event.requested_by}: {event.reason}"
            )
        
        except Exception as e:
            self.logger.error(
                f"[EventHandlers] Error handling manual_retrain_request: {e}",
                exc_info=True
            )


def setup_event_subscriptions(event_bus, handlers):
    """
    Setup event subscriptions.
    
    Args:
        event_bus: EventBus instance
        handlers: EventHandlers instance
    """
    event_bus.subscribe(
        "performance.metrics_updated",
        handlers.handle_performance_metrics_updated
    )
    
    event_bus.subscribe(
        "data.drift_signal",
        handlers.handle_data_drift_signal
    )
    
    event_bus.subscribe(
        "manual.retrain_request",
        handlers.handle_manual_retrain_request
    )
    
    logger.info("[EventHandlers] Event subscriptions setup complete")
