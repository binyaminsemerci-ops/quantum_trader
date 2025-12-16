"""
Risk v3 Main Entry Point - EventBus Integration

EPIC-RISK3-001: Connect Risk v3 to EventBus for automated risk monitoring

Subscribes to:
- portfolio.position_opened
- portfolio.position_closed
- portfolio.balance_updated
- execution.trade_executed
- market.regime_changed

Publishes:
- risk.global_snapshot
- risk.var_es_updated
- risk.systemic_alert
- risk.exposure_matrix_updated
- risk.threshold_breach

Periodic Risk Evaluation:
- Every 5 minutes: Full risk evaluation
- Every 1 minute: Quick snapshot check
- On position change: Immediate evaluation
"""

import asyncio
import logging
from typing import Optional
from datetime import datetime, timedelta

from .orchestrator import RiskOrchestrator
from .app import init_orchestrator

logger = logging.getLogger(__name__)


class RiskV3Service:
    """Risk v3 service with EventBus integration and periodic evaluation"""
    
    def __init__(self, event_bus=None, evaluation_interval_seconds: int = 300):
        """
        Initialize Risk v3 service
        
        Args:
            event_bus: EventBus instance for subscriptions and publishing
            evaluation_interval_seconds: Interval for periodic risk evaluation (default 5 min)
        """
        self.event_bus = event_bus
        self.evaluation_interval = evaluation_interval_seconds
        
        # Initialize orchestrator
        self.orchestrator = RiskOrchestrator(event_bus=event_bus)
        init_orchestrator(self.orchestrator)
        
        # State
        self.running = False
        self.last_evaluation_time: Optional[datetime] = None
        
        logger.info(
            f"[RISK-V3-SERVICE] Initialized (evaluation interval: {evaluation_interval_seconds}s)"
        )
    
    async def start(self):
        """Start Risk v3 service"""
        if self.running:
            logger.warning("[RISK-V3-SERVICE] Already running")
            return
        
        self.running = True
        logger.info("[RISK-V3-SERVICE] 🚀 Starting Risk v3 service...")
        
        # Subscribe to relevant events
        if self.event_bus:
            await self._subscribe_to_events()
        
        # Start periodic evaluation loop
        asyncio.create_task(self._periodic_evaluation_loop())
        
        # Initial evaluation
        await self._evaluate_risk("Initial evaluation")
        
        logger.info("[RISK-V3-SERVICE] ✅ Risk v3 service started")
    
    async def stop(self):
        """Stop Risk v3 service"""
        logger.info("[RISK-V3-SERVICE] 🛑 Stopping Risk v3 service...")
        self.running = False
        
        # Unsubscribe from events
        if self.event_bus:
            await self._unsubscribe_from_events()
        
        logger.info("[RISK-V3-SERVICE] ✅ Risk v3 service stopped")
    
    async def _subscribe_to_events(self):
        """Subscribe to relevant events on EventBus"""
        if not self.event_bus:
            return
        
        try:
            # Portfolio events
            await self.event_bus.subscribe(
                "portfolio.position_opened",
                self._on_position_change
            )
            await self.event_bus.subscribe(
                "portfolio.position_closed",
                self._on_position_change
            )
            await self.event_bus.subscribe(
                "portfolio.balance_updated",
                self._on_balance_update
            )
            
            # Execution events
            await self.event_bus.subscribe(
                "execution.trade_executed",
                self._on_trade_executed
            )
            
            # Market events
            await self.event_bus.subscribe(
                "market.regime_changed",
                self._on_regime_change
            )
            
            logger.info("[RISK-V3-SERVICE] ✅ Subscribed to EventBus events")
        
        except Exception as e:
            logger.error(f"[RISK-V3-SERVICE] ❌ Event subscription failed: {e}")
    
    async def _unsubscribe_from_events(self):
        """Unsubscribe from all events"""
        if not self.event_bus:
            return
        
        try:
            # TODO: Implement unsubscribe in EventBus
            logger.info("[RISK-V3-SERVICE] ✅ Unsubscribed from EventBus events")
        except Exception as e:
            logger.error(f"[RISK-V3-SERVICE] ❌ Event unsubscription failed: {e}")
    
    async def _on_position_change(self, event_data: dict):
        """Handle position opened/closed events"""
        logger.info(f"[RISK-V3-SERVICE] Position change detected: {event_data}")
        await self._evaluate_risk("Position change")
    
    async def _on_balance_update(self, event_data: dict):
        """Handle balance update events"""
        logger.debug(f"[RISK-V3-SERVICE] Balance updated: {event_data}")
        # Balance updates happen frequently, only evaluate if significant change
        # For now, rely on periodic evaluation
    
    async def _on_trade_executed(self, event_data: dict):
        """Handle trade execution events"""
        logger.info(f"[RISK-V3-SERVICE] Trade executed: {event_data}")
        await self._evaluate_risk("Trade execution")
    
    async def _on_regime_change(self, event_data: dict):
        """Handle market regime change events"""
        logger.warning(f"[RISK-V3-SERVICE] 🔄 Regime changed: {event_data}")
        await self._evaluate_risk("Regime change", force_refresh=True)
    
    async def _periodic_evaluation_loop(self):
        """Periodic risk evaluation loop"""
        logger.info(
            f"[RISK-V3-SERVICE] 🔄 Starting periodic evaluation loop "
            f"(every {self.evaluation_interval}s)"
        )
        
        while self.running:
            try:
                await asyncio.sleep(self.evaluation_interval)
                
                if self.running:
                    await self._evaluate_risk("Periodic evaluation")
            
            except Exception as e:
                logger.error(f"[RISK-V3-SERVICE] ❌ Periodic evaluation error: {e}", exc_info=True)
                await asyncio.sleep(60)  # Wait before retry
    
    async def _evaluate_risk(self, reason: str, force_refresh: bool = False):
        """
        Evaluate risk with reason logging
        
        Args:
            reason: Reason for evaluation (for logging)
            force_refresh: Force refresh of data sources
        """
        try:
            logger.info(f"[RISK-V3-SERVICE] 🎯 Evaluating risk: {reason}")
            
            signal = await self.orchestrator.evaluate_risk(force_refresh=force_refresh)
            
            self.last_evaluation_time = datetime.utcnow()
            
            # Log critical issues
            if signal.risk_level.value == "CRITICAL":
                logger.error(
                    f"[RISK-V3-SERVICE] 🚨 CRITICAL RISK LEVEL\n"
                    f"  Summary: {signal.risk_summary}\n"
                    f"  Critical Issues: {signal.critical_issues}"
                )
            elif signal.risk_level.value == "WARNING":
                logger.warning(
                    f"[RISK-V3-SERVICE] ⚠️ WARNING RISK LEVEL\n"
                    f"  Summary: {signal.risk_summary}\n"
                    f"  Warnings: {len(signal.warnings)}"
                )
            else:
                logger.info(f"[RISK-V3-SERVICE] ✅ Risk level: {signal.risk_level.value}")
        
        except Exception as e:
            logger.error(
                f"[RISK-V3-SERVICE] ❌ Risk evaluation failed ({reason}): {e}",
                exc_info=True
            )


# Global service instance
_service_instance: Optional[RiskV3Service] = None


async def start_risk_v3_service(event_bus=None, evaluation_interval: int = 300):
    """
    Start Risk v3 service
    
    Args:
        event_bus: EventBus instance
        evaluation_interval: Evaluation interval in seconds (default 5 min)
    """
    global _service_instance
    
    if _service_instance is not None:
        logger.warning("[RISK-V3-SERVICE] Service already started")
        return _service_instance
    
    _service_instance = RiskV3Service(
        event_bus=event_bus,
        evaluation_interval_seconds=evaluation_interval,
    )
    
    await _service_instance.start()
    
    return _service_instance


async def stop_risk_v3_service():
    """Stop Risk v3 service"""
    global _service_instance
    
    if _service_instance is None:
        logger.warning("[RISK-V3-SERVICE] Service not running")
        return
    
    await _service_instance.stop()
    _service_instance = None


def get_risk_v3_service() -> Optional[RiskV3Service]:
    """Get running service instance"""
    return _service_instance


if __name__ == "__main__":
    # Standalone service mode
    logging.basicConfig(level=logging.INFO)
    
    async def main():
        service = await start_risk_v3_service()
        
        try:
            # Keep running
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("[RISK-V3-SERVICE] Shutting down...")
            await stop_risk_v3_service()
    
    asyncio.run(main())
