"""
EventBus Usage Examples and Integration Guide

Demonstrates how to integrate EventBus into Quantum Trader modules.

Examples:
1. Basic usage with async handlers
2. MSC AI publishing policy updates
3. System Health Monitor publishing alerts
4. Logger subscribing to all events
5. Discord notifier for critical alerts
6. Opportunity Ranker publishing rankings
7. Full system integration

Run: python backend/services/demo_event_bus.py
"""

import asyncio
import logging
from datetime import datetime

from backend.services.event_bus import (
    Event,
    PolicyUpdatedEvent,
    StrategyPromotedEvent,
    ModelPromotedEvent,
    HealthStatusChangedEvent,
    OpportunitiesUpdatedEvent,
    TradeExecutedEvent,
    InMemoryEventBus,
    create_event_bus,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Example 1: Basic Usage
# ============================================================================

async def example_1_basic_usage():
    """Basic EventBus usage with async handlers."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Usage")
    print("="*70)
    
    bus = InMemoryEventBus()
    
    # Define a simple handler
    async def on_policy_updated(event: Event):
        print(f"✅ Policy updated: {event.payload}")
    
    # Subscribe
    bus.subscribe("policy.updated", on_policy_updated)
    
    # Start event loop in background
    loop_task = asyncio.create_task(bus.run_forever())
    
    # Publish event
    await bus.publish(PolicyUpdatedEvent.create(
        risk_mode="AGGRESSIVE",
        max_positions=15,
        global_min_confidence=0.7
    ))
    
    # Wait for processing
    await asyncio.sleep(0.1)
    
    # Shutdown
    await bus.shutdown()
    await loop_task
    
    print(f"📊 Stats: {bus.get_stats()}")


# ============================================================================
# Example 2: MSC AI Integration
# ============================================================================

class MockMSCAI:
    """Mock Meta Strategy Controller AI."""
    
    def __init__(self, event_bus: InMemoryEventBus):
        self.event_bus = event_bus
        self.policy_store = {}  # Simulated PolicyStore
    
    async def update_risk_mode(self, new_mode: str, reason: str):
        """Update risk mode and publish event."""
        old_mode = self.policy_store.get("risk_mode", "NORMAL")
        
        # Update policy
        self.policy_store["risk_mode"] = new_mode
        
        logger.info(
            "🧠 [MSC AI] Risk mode changed: %s → %s (reason: %s)",
            old_mode,
            new_mode,
            reason
        )
        
        # Publish event
        await self.event_bus.publish(PolicyUpdatedEvent.create(
            risk_mode=new_mode,
            changes={
                "risk_mode": f"{old_mode} → {new_mode}",
                "reason": reason
            }
        ))


async def example_2_msc_ai_integration():
    """MSC AI publishing policy updates."""
    print("\n" + "="*70)
    print("EXAMPLE 2: MSC AI Integration")
    print("="*70)
    
    bus = InMemoryEventBus()
    msc_ai = MockMSCAI(bus)
    
    # Subscribe to policy updates
    async def on_policy_updated(event: Event):
        print(f"📢 Policy update received: {event.payload}")
    
    bus.subscribe("policy.updated", on_policy_updated)
    
    # Start event loop
    loop_task = asyncio.create_task(bus.run_forever())
    
    # MSC AI makes decisions
    await msc_ai.update_risk_mode("DEFENSIVE", "High market volatility detected")
    await asyncio.sleep(0.1)
    
    await msc_ai.update_risk_mode("AGGRESSIVE", "Strong trend confirmed")
    await asyncio.sleep(0.1)
    
    # Shutdown
    await bus.shutdown()
    await loop_task
    
    print(f"📊 Stats: {bus.get_stats()}")


# ============================================================================
# Example 3: System Health Monitor Integration
# ============================================================================

class MockSystemHealthMonitor:
    """Mock System Health Monitor."""
    
    def __init__(self, event_bus: InMemoryEventBus):
        self.event_bus = event_bus
        self.current_status = "HEALTHY"
    
    async def check_health_and_publish(self):
        """Check system health and publish if status changed."""
        # Simulate health check
        new_status = "CRITICAL"
        failed_modules = ["policy_store", "redis"]
        
        if new_status != self.current_status:
            logger.warning(
                "🏥 [Health Monitor] Status changed: %s → %s",
                self.current_status,
                new_status
            )
            
            # Publish event
            await self.event_bus.publish(HealthStatusChangedEvent.create(
                status=new_status,
                previous_status=self.current_status,
                failed_modules=failed_modules,
                details={"timestamp": datetime.utcnow().isoformat()}
            ))
            
            self.current_status = new_status


async def example_3_health_monitor_integration():
    """System Health Monitor publishing alerts."""
    print("\n" + "="*70)
    print("EXAMPLE 3: System Health Monitor Integration")
    print("="*70)
    
    bus = InMemoryEventBus()
    health_monitor = MockSystemHealthMonitor(bus)
    
    # Subscribe to health alerts
    async def on_health_changed(event: Event):
        status = event.payload["status"]
        failed = event.payload["failed_modules"]
        print(f"🚨 ALERT: System health is now {status}")
        print(f"   Failed modules: {', '.join(failed)}")
    
    bus.subscribe("health.status_changed", on_health_changed)
    
    # Start event loop
    loop_task = asyncio.create_task(bus.run_forever())
    
    # Health monitor detects issue
    await health_monitor.check_health_and_publish()
    await asyncio.sleep(0.1)
    
    # Shutdown
    await bus.shutdown()
    await loop_task


# ============================================================================
# Example 4: Logger Subscribing to All Events
# ============================================================================

class EventLogger:
    """Subscribes to all event types and logs them."""
    
    def __init__(self, event_bus: InMemoryEventBus):
        self.event_bus = event_bus
        self.logged_events = []
        self._register_handlers()
    
    def _register_handlers(self):
        """Register handlers for all event types."""
        event_types = [
            "policy.updated",
            "strategy.promoted",
            "model.promoted",
            "health.status_changed",
            "opportunities.updated",
            "trade.executed",
        ]
        
        for event_type in event_types:
            self.event_bus.subscribe(event_type, self.log_event)
    
    async def log_event(self, event: Event):
        """Log any event."""
        self.logged_events.append(event)
        logger.info(
            "📝 [EventLogger] type=%s source=%s payload=%s",
            event.type,
            event.source,
            event.payload
        )


async def example_4_logger_subscription():
    """Logger subscribing to all events."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Logger Subscribing to All Events")
    print("="*70)
    
    bus = InMemoryEventBus()
    event_logger = EventLogger(bus)
    
    # Start event loop
    loop_task = asyncio.create_task(bus.run_forever())
    
    # Publish various events
    await bus.publish(PolicyUpdatedEvent.create(risk_mode="NORMAL"))
    await bus.publish(StrategyPromotedEvent.create(
        strategy_id="strat-1",
        strategy_name="MeanReversion",
        from_state="SHADOW",
        to_state="LIVE"
    ))
    await bus.publish(TradeExecutedEvent.create(
        symbol="BTCUSDT",
        side="BUY",
        quantity=0.1,
        price=45000.0,
        order_id="order-123"
    ))
    
    # Wait for processing
    await asyncio.sleep(0.2)
    
    # Shutdown
    await bus.shutdown()
    await loop_task
    
    print(f"\n📊 EventLogger captured {len(event_logger.logged_events)} events")


# ============================================================================
# Example 5: Discord Notifier for Critical Alerts
# ============================================================================

class DiscordNotifier:
    """Sends critical alerts to Discord."""
    
    def __init__(self, event_bus: InMemoryEventBus, webhook_url: str = None):
        self.event_bus = event_bus
        self.webhook_url = webhook_url
        self._register_handlers()
    
    def _register_handlers(self):
        """Subscribe to critical events."""
        # Only care about health alerts and big trades
        self.event_bus.subscribe("health.status_changed", self.on_health_alert)
        self.event_bus.subscribe("trade.executed", self.on_trade_executed)
    
    async def on_health_alert(self, event: Event):
        """React to health status changes."""
        status = event.payload.get("status")
        
        if status == "CRITICAL":
            await self._send_discord_message(
                f"🚨 CRITICAL ALERT: System health is CRITICAL!\n"
                f"Failed modules: {event.payload.get('failed_modules')}"
            )
    
    async def on_trade_executed(self, event: Event):
        """React to large trades."""
        pnl = event.payload.get("pnl")
        
        if pnl and abs(pnl) > 1000:
            symbol = event.payload["symbol"]
            side = event.payload["side"]
            await self._send_discord_message(
                f"💰 Large trade: {side} {symbol} | PnL: ${pnl:,.2f}"
            )
    
    async def _send_discord_message(self, message: str):
        """Send message to Discord (simulated)."""
        print(f"🔔 [Discord] {message}")
        # In production: HTTP POST to webhook_url


async def example_5_discord_notifier():
    """Discord notifier for critical alerts."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Discord Notifier for Critical Alerts")
    print("="*70)
    
    bus = InMemoryEventBus()
    notifier = DiscordNotifier(bus)
    
    # Start event loop
    loop_task = asyncio.create_task(bus.run_forever())
    
    # Publish various events
    await bus.publish(HealthStatusChangedEvent.create(
        status="CRITICAL",
        previous_status="WARNING",
        failed_modules=["redis", "database"]
    ))
    
    await bus.publish(TradeExecutedEvent.create(
        symbol="BTCUSDT",
        side="SELL",
        quantity=1.0,
        price=45000.0,
        order_id="order-456",
        pnl=1500.0  # Large profit
    ))
    
    await bus.publish(TradeExecutedEvent.create(
        symbol="ETHUSDT",
        side="BUY",
        quantity=10.0,
        price=2500.0,
        order_id="order-789",
        pnl=50.0  # Small profit - should NOT trigger Discord
    ))
    
    # Wait for processing
    await asyncio.sleep(0.2)
    
    # Shutdown
    await bus.shutdown()
    await loop_task


# ============================================================================
# Example 6: Opportunity Ranker Integration
# ============================================================================

class MockOpportunityRanker:
    """Mock Opportunity Ranker."""
    
    def __init__(self, event_bus: InMemoryEventBus):
        self.event_bus = event_bus
    
    async def compute_and_publish_rankings(self):
        """Compute symbol rankings and publish."""
        # Simulate ranking computation
        rankings = {
            "BTCUSDT": 0.95,
            "ETHUSDT": 0.88,
            "SOLUSDT": 0.82,
            "ADAUSDT": 0.75,
            "DOTUSDT": 0.70,
        }
        
        top_symbols = sorted(rankings.keys(), key=lambda k: rankings[k], reverse=True)[:3]
        
        logger.info("📊 [OppRank] Computed rankings for %d symbols", len(rankings))
        
        # Publish event
        await self.event_bus.publish(OpportunitiesUpdatedEvent.create(
            top_symbols=top_symbols,
            rankings=rankings,
            num_symbols_scored=len(rankings),
            top_score=max(rankings.values())
        ))


async def example_6_opportunity_ranker():
    """Opportunity Ranker publishing rankings."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Opportunity Ranker Integration")
    print("="*70)
    
    bus = InMemoryEventBus()
    opp_ranker = MockOpportunityRanker(bus)
    
    # Subscribe to opportunity updates
    async def on_opportunities_updated(event: Event):
        top_symbols = event.payload["top_symbols"]
        rankings = event.payload["rankings"]
        
        print(f"🎯 Top opportunities:")
        for symbol in top_symbols:
            score = rankings[symbol]
            print(f"   {symbol}: {score:.2f}")
    
    bus.subscribe("opportunities.updated", on_opportunities_updated)
    
    # Start event loop
    loop_task = asyncio.create_task(bus.run_forever())
    
    # Ranker computes and publishes
    await opp_ranker.compute_and_publish_rankings()
    await asyncio.sleep(0.1)
    
    # Shutdown
    await bus.shutdown()
    await loop_task


# ============================================================================
# Example 7: Full System Integration
# ============================================================================

async def example_7_full_integration():
    """Complete system integration example."""
    print("\n" + "="*70)
    print("EXAMPLE 7: Full System Integration")
    print("="*70)
    
    # Create EventBus
    bus = create_event_bus()
    
    # Create all components
    msc_ai = MockMSCAI(bus)
    health_monitor = MockSystemHealthMonitor(bus)
    opp_ranker = MockOpportunityRanker(bus)
    event_logger = EventLogger(bus)
    notifier = DiscordNotifier(bus)
    
    # Start event loop
    loop_task = asyncio.create_task(bus.run_forever())
    
    print("\n🚀 Quantum Trader EventBus Running...\n")
    
    # Simulate system activity
    print("1️⃣ MSC AI updates policy...")
    await msc_ai.update_risk_mode("DEFENSIVE", "Market volatility increased")
    await asyncio.sleep(0.1)
    
    print("\n2️⃣ Opportunity Ranker computes rankings...")
    await opp_ranker.compute_and_publish_rankings()
    await asyncio.sleep(0.1)
    
    print("\n3️⃣ Health Monitor detects issue...")
    await health_monitor.check_health_and_publish()
    await asyncio.sleep(0.1)
    
    print("\n4️⃣ Strategy promoted...")
    await bus.publish(StrategyPromotedEvent.create(
        strategy_id="strat-123",
        strategy_name="TrendFollowing_v2",
        from_state="SHADOW",
        to_state="LIVE",
        performance_score=0.92,
        reason="Consistently outperformed baseline"
    ))
    await asyncio.sleep(0.1)
    
    print("\n5️⃣ Large trade executed...")
    await bus.publish(TradeExecutedEvent.create(
        symbol="BTCUSDT",
        side="SELL",
        quantity=2.0,
        price=46000.0,
        order_id="order-999",
        strategy_id="strat-123",
        pnl=2500.0
    ))
    await asyncio.sleep(0.1)
    
    # Shutdown
    print("\n🛑 Shutting down EventBus...")
    await bus.shutdown()
    await loop_task
    
    # Final stats
    print("\n" + "="*70)
    print("📊 Final Statistics:")
    stats = bus.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    print("="*70)


# ============================================================================
# Main
# ============================================================================

async def main():
    """Run all examples."""
    await example_1_basic_usage()
    await example_2_msc_ai_integration()
    await example_3_health_monitor_integration()
    await example_4_logger_subscription()
    await example_5_discord_notifier()
    await example_6_opportunity_ranker()
    await example_7_full_integration()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("EventBus Usage Examples for Quantum Trader")
    print("="*70)
    
    asyncio.run(main())
    
    print("\n✅ All examples completed successfully!")
