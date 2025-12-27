"""
Example usage of the EventBus subsystem.

This demonstrates how various Quantum Trader components would use the EventBus
for decoupled communication and system coordination.
"""

import asyncio
import logging
from datetime import datetime

from backend.services.eventbus import (
    InMemoryEventBus,
    Event,
    PolicyUpdatedEvent,
    StrategyPromotedEvent,
    ModelPromotedEvent,
    HealthStatusChangedEvent,
    OpportunitiesUpdatedEvent,
    TradeExecutedEvent,
    RiskMode,
    StrategyLifecycle,
    HealthStatus,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Example 1: Simple event handler
# ============================================================================

async def simple_logger(event: Event) -> None:
    """Simple async handler that logs all events."""
    logger.info(f"📨 Event received: {event.type} at {event.timestamp}")
    logger.info(f"   Payload: {event.payload}")


# ============================================================================
# Example 2: MSC AI publishes policy updates
# ============================================================================

class MockMSCAI:
    """Mock Meta Strategy Controller that publishes policy updates."""
    
    def __init__(self, event_bus: InMemoryEventBus):
        self.event_bus = event_bus
    
    async def update_risk_mode(self, new_mode: RiskMode) -> None:
        """
        Change risk mode and notify the system.
        
        In reality, this would:
        1. Analyze market conditions
        2. Update PolicyStore
        3. Publish event so other components can react
        """
        logger.info(f"🧠 MSC AI: Changing risk mode to {new_mode.value}")
        
        # Create and publish event
        event = PolicyUpdatedEvent.create(
            risk_mode=new_mode,
            allowed_strategies=["momentum_v2", "mean_reversion_v1"],
            global_min_confidence=0.7 if new_mode == RiskMode.DEFENSIVE else 0.6,
            max_risk_per_trade=0.01 if new_mode == RiskMode.DEFENSIVE else 0.02,
            max_positions=3 if new_mode == RiskMode.DEFENSIVE else 5,
        )
        
        await self.event_bus.publish(event)


# ============================================================================
# Example 3: Orchestrator reacts to policy changes
# ============================================================================

class MockOrchestrator:
    """Mock Orchestrator that listens for policy updates."""
    
    def __init__(self, event_bus: InMemoryEventBus):
        self.event_bus = event_bus
        self.current_policy = {}
        
        # Subscribe to policy updates
        event_bus.subscribe("policy.updated", self.on_policy_updated)
    
    async def on_policy_updated(self, event: Event) -> None:
        """Handle policy update events."""
        logger.info(f"🎯 Orchestrator: New policy received!")
        self.current_policy = event.payload
        
        # In reality, would reload internal state
        logger.info(f"   → Risk mode: {event.payload['risk_mode']}")
        logger.info(f"   → Max positions: {event.payload['max_positions']}")
        logger.info(f"   → Min confidence: {event.payload['global_min_confidence']}")


# ============================================================================
# Example 4: SG AI publishes strategy promotions
# ============================================================================

class MockStrategyGenerator:
    """Mock Strategy Generator AI that promotes strategies."""
    
    def __init__(self, event_bus: InMemoryEventBus):
        self.event_bus = event_bus
    
    async def promote_strategy(self, strategy_id: str) -> None:
        """Promote a strategy from SHADOW to LIVE."""
        logger.info(f"🚀 SG AI: Promoting strategy {strategy_id} to LIVE")
        
        event = StrategyPromotedEvent.create(
            strategy_id=strategy_id,
            from_stage=StrategyLifecycle.SHADOW,
            to_stage=StrategyLifecycle.LIVE,
            reason="Shadow performance exceeded thresholds",
            metrics={
                "sharpe_ratio": 2.3,
                "win_rate": 0.65,
                "avg_profit": 1.8,
            }
        )
        
        await self.event_bus.publish(event)


# ============================================================================
# Example 5: Health Monitor publishes critical alerts
# ============================================================================

class MockHealthMonitor:
    """Mock System Health Monitor that detects issues."""
    
    def __init__(self, event_bus: InMemoryEventBus):
        self.event_bus = event_bus
        self.current_status = HealthStatus.HEALTHY
    
    async def check_health(self) -> None:
        """Simulate health check that detects a problem."""
        # Simulate detecting an issue
        new_status = HealthStatus.CRITICAL
        
        if new_status != self.current_status:
            logger.warning(f"⚠️  Health Monitor: Status changed to {new_status.value}")
            
            event = HealthStatusChangedEvent.create(
                old_status=self.current_status,
                new_status=new_status,
                component="DrawdownGuard",
                reason="Daily drawdown exceeded 5%",
                metrics={
                    "current_dd": -5.2,
                    "max_allowed_dd": -5.0,
                    "equity": 95000,
                }
            )
            
            await self.event_bus.publish(event)
            self.current_status = new_status


# ============================================================================
# Example 6: Discord notifier reacts to critical health
# ============================================================================

class MockDiscordNotifier:
    """Mock Discord integration that sends alerts."""
    
    def __init__(self, event_bus: InMemoryEventBus):
        self.event_bus = event_bus
        
        # Subscribe to health changes
        event_bus.subscribe("health.status_changed", self.on_health_changed)
    
    async def on_health_changed(self, event: Event) -> None:
        """Send Discord alert on critical health."""
        new_status = event.payload["new_status"]
        
        if new_status == "CRITICAL":
            logger.error(f"🚨 Discord: CRITICAL ALERT!")
            logger.error(f"   Component: {event.payload['component']}")
            logger.error(f"   Reason: {event.payload['reason']}")
            # In reality: await discord_client.send_message(...)


# ============================================================================
# Example 7: Sync handler (runs in thread pool)
# ============================================================================

def sync_metrics_logger(event: Event) -> None:
    """
    Synchronous handler example.
    
    The EventBus will automatically run this in a thread pool
    to avoid blocking the async loop.
    """
    if event.type == "trade.executed":
        logger.info(f"💰 Metrics Logger: Trade executed on {event.payload['symbol']}")


# ============================================================================
# Main example runner
# ============================================================================

async def main():
    """Run comprehensive EventBus example."""
    logger.info("=" * 70)
    logger.info("🎬 Starting EventBus Example")
    logger.info("=" * 70)
    
    # 1. Create EventBus
    bus = InMemoryEventBus(max_queue_size=1000)
    
    # 2. Start the event loop in background
    bus_task = asyncio.create_task(bus.run_forever())
    
    # 3. Create mock components
    msc_ai = MockMSCAI(bus)
    orchestrator = MockOrchestrator(bus)
    sg_ai = MockStrategyGenerator(bus)
    health_monitor = MockHealthMonitor(bus)
    discord_notifier = MockDiscordNotifier(bus)
    
    # 4. Register additional handlers
    bus.subscribe("policy.updated", simple_logger)
    bus.subscribe("strategy.promoted", simple_logger)
    bus.subscribe("trade.executed", sync_metrics_logger)
    
    logger.info("\n✅ EventBus initialized with all components\n")
    
    # 5. Simulate system events
    
    # Scenario 1: MSC AI changes risk mode
    logger.info("\n" + "─" * 70)
    logger.info("📋 SCENARIO 1: MSC AI updates policy to DEFENSIVE mode")
    logger.info("─" * 70)
    await msc_ai.update_risk_mode(RiskMode.DEFENSIVE)
    await asyncio.sleep(0.1)  # Allow handlers to process
    
    # Scenario 2: SG AI promotes a strategy
    logger.info("\n" + "─" * 70)
    logger.info("📋 SCENARIO 2: SG AI promotes strategy to LIVE")
    logger.info("─" * 70)
    await sg_ai.promote_strategy("momentum_enhanced_v3")
    await asyncio.sleep(0.1)
    
    # Scenario 3: Health issue detected
    logger.info("\n" + "─" * 70)
    logger.info("📋 SCENARIO 3: Health Monitor detects critical issue")
    logger.info("─" * 70)
    await health_monitor.check_health()
    await asyncio.sleep(0.1)
    
    # Scenario 4: Multiple events published at once
    logger.info("\n" + "─" * 70)
    logger.info("📋 SCENARIO 4: High-volume event publishing")
    logger.info("─" * 70)
    
    # Publish opportunities update
    opp_event = OpportunitiesUpdatedEvent.create(
        top_symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        scores={"BTCUSDT": 0.92, "ETHUSDT": 0.88, "SOLUSDT": 0.85},
        criteria={"min_volume": 1e9, "trend_strength": 0.7},
        excluded_count=47,
    )
    await bus.publish(opp_event)
    
    # Publish trade execution
    trade_event = TradeExecutedEvent.create(
        order_id="ORD_12345",
        symbol="BTCUSDT",
        side="BUY",
        size=0.05,
        price=42350.50,
        strategy_id="momentum_enhanced_v3",
        model="ensemble_v1",
    )
    await bus.publish(trade_event)
    
    # Publish model promotion
    model_event = ModelPromotedEvent.create(
        model_name="XGBoost",
        old_version="v1.2.3",
        new_version="v1.3.0",
        metrics={"accuracy": 0.72, "sharpe": 1.8},
        shadow_performance={"win_rate": 0.68, "avg_profit": 2.1},
    )
    await bus.publish(model_event)
    
    await asyncio.sleep(0.2)  # Allow all events to process
    
    # 6. Show statistics
    logger.info("\n" + "=" * 70)
    logger.info("📊 EventBus Statistics")
    logger.info("=" * 70)
    stats = bus.get_stats()
    for key, value in stats.items():
        logger.info(f"   {key}: {value}")
    
    # 7. Cleanup
    logger.info("\n🛑 Stopping EventBus...")
    bus.stop()
    bus_task.cancel()
    
    try:
        await bus_task
    except asyncio.CancelledError:
        pass
    
    logger.info("✅ Example completed successfully!\n")


if __name__ == "__main__":
    asyncio.run(main())
