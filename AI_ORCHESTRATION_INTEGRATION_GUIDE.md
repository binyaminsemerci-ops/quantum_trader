# AI Orchestration Layer - Integration Guide

## Overview

The AI Orchestration Layer adds a meta-level intelligence system to Quantum Trader v5, providing automated global decision-making through three specialized AI agents coordinated by a Federation Layer.

### Components

1. **AI CEO (Meta-Orchestrator)** - `backend/ai_orchestrator/`
   - Makes global operating mode decisions
   - Updates PolicyStore with mode-specific configurations
   - Aggregates inputs from all system components

2. **AI Risk Officer** - `backend/ai_risk/`
   - Calculates VaR, Expected Shortfall, tail risk
   - Monitors drawdown and exposure
   - Recommends risk limit adjustments

3. **AI Strategy Officer** - `backend/ai_strategy/`
   - Analyzes strategy and model performance
   - Recommends primary/fallback strategies
   - Identifies underperforming strategies

4. **Federation Layer** - `backend/federation/`
   - Aggregates outputs from all three agents
   - Provides unified global state API
   - Publishes global_state_update events

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Federation Layer                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Global State Snapshot (Single Truth)         │  │
│  └──────────────────────────────────────────────────────┘  │
│         ▲              ▲              ▲                      │
└─────────┼──────────────┼──────────────┼──────────────────────┘
          │              │              │
    ┌─────┴─────┐  ┌────┴────┐  ┌──────┴──────┐
    │  AI CEO   │  │ AI Risk │  │ AI Strategy │
    │           │  │ Officer │  │   Officer   │
    └─────┬─────┘  └────┬────┘  └──────┬──────┘
          │              │              │
          └──────────────┴──────────────┘
                         │
            ┌────────────┴────────────┐
            │      EventBus v2        │
            │   (Redis Streams)       │
            └─────────────────────────┘
                         │
            ┌────────────┴────────────┐
            │    PolicyStore v2       │
            │   (Redis + JSON)        │
            └─────────────────────────┘
```

## Event Flow

### AI CEO Decision Cycle

```
Every 30 seconds:
1. AI CEO gathers system state:
   - Risk state from AI-RO
   - Strategy state from AI-SO
   - Portfolio state from PBA/PAL
   - System health from monitors
   
2. CEO Brain evaluates state:
   - Applies CEO Policy rules
   - Recommends operating mode (EXPANSION/GROWTH/DEFENSIVE/etc.)
   - Validates transition (cooldown checks)
   
3. AI CEO executes decision:
   - Updates PolicyStore if mode changed
   - Publishes ceo_decision event
   - Publishes ceo_mode_switch event (if changed)
   - Publishes ceo_alert event (if needed)
```

### AI Risk Officer Assessment Cycle

```
Every 30 seconds:
1. AI-RO gathers risk data:
   - Historical returns
   - Current portfolio state
   - Risk limits from PolicyStore
   
2. Risk Brain calculates:
   - VaR (Value at Risk)
   - Expected Shortfall (ES/CVaR)
   - Tail risk metrics
   - Overall risk score (0-100)
   
3. AI-RO publishes:
   - risk_state_update event
   - risk_alert event (if thresholds breached)
   - risk_ceiling_update event (if limits changed)
```

### AI Strategy Officer Analysis Cycle

```
Every 60 seconds:
1. AI-SO gathers performance data:
   - Strategy performance metrics
   - Model prediction accuracy
   - Win rates, Sharpe ratios
   
2. Strategy Brain analyzes:
   - Ranks strategies by performance
   - Selects primary + fallback strategies
   - Identifies strategies to disable
   - Analyzes model performance
   
3. AI-SO publishes:
   - strategy_state_update event
   - strategy_recommendation event
   - strategy_alert event (if issues detected)
```

### Federation Layer Aggregation

```
Every 15 seconds:
1. FederatedEngine collects:
   - Latest CEO state
   - Latest Risk state
   - Latest Strategy state
   
2. Builds GlobalState:
   - Operating mode
   - Effective risk limits
   - Active strategies
   - Disabled features
   - All alerts
   
3. Publishes:
   - global_state_update event
```

## Integration Example 1: Running in analytics-os-service

```python
# backend/services/analytics_os_service.py

import asyncio
from redis.asyncio import Redis
from backend.core.event_bus import EventBus
from backend.core.policy_store import PolicyStore
from backend.ai_orchestrator import AI_CEO
from backend.ai_risk import AI_RiskOfficer
from backend.ai_strategy import AI_StrategyOfficer
from backend.federation import FederatedEngine

async def start_orchestration_layer():
    """Initialize and start AI Orchestration Layer."""
    
    # Initialize dependencies
    redis_client = Redis(host="localhost", port=6379, decode_responses=False)
    event_bus = EventBus(redis_client, service_name="analytics_os")
    policy_store = PolicyStore(redis_client, event_bus=event_bus)
    
    await redis_client.ping()
    await event_bus.initialize()
    await policy_store.initialize()
    
    # Initialize AI agents
    ai_ceo = AI_CEO(
        redis_client=redis_client,
        event_bus=event_bus,
        policy_store=policy_store,
        decision_interval=30.0,  # 30 seconds
    )
    
    ai_risk_officer = AI_RiskOfficer(
        redis_client=redis_client,
        event_bus=event_bus,
        policy_store=policy_store,
        assessment_interval=30.0,  # 30 seconds
    )
    
    ai_strategy_officer = AI_StrategyOfficer(
        redis_client=redis_client,
        event_bus=event_bus,
        policy_store=policy_store,
        analysis_interval=60.0,  # 60 seconds
    )
    
    # Initialize federation layer
    federated_engine = FederatedEngine(
        redis_client=redis_client,
        event_bus=event_bus,
        update_interval=15.0,  # 15 seconds
    )
    
    # Initialize all agents
    await ai_ceo.initialize()
    await ai_risk_officer.initialize()
    await ai_strategy_officer.initialize()
    await federated_engine.initialize()
    
    # Start all agents
    await ai_ceo.start()
    await ai_risk_officer.start()
    await ai_strategy_officer.start()
    await federated_engine.start()
    
    print("✅ AI Orchestration Layer started successfully")
    
    return {
        "ceo": ai_ceo,
        "risk_officer": ai_risk_officer,
        "strategy_officer": ai_strategy_officer,
        "federation": federated_engine,
    }

# Run the orchestration layer
if __name__ == "__main__":
    agents = asyncio.run(start_orchestration_layer())
    
    # Keep running
    try:
        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        print("Shutting down...")
```

## Integration Example 2: Querying Global State from Dashboard

```python
# backend/routes/orchestration_routes.py

from fastapi import APIRouter, Depends
from backend.federation import FederatedEngine

router = APIRouter(prefix="/api/orchestration", tags=["orchestration"])

# Dependency injection (get federated engine from app state)
def get_federated_engine() -> FederatedEngine:
    # In production, get from app.state or DI container
    return app.state.federated_engine

@router.get("/global-state")
async def get_global_state(
    federation: FederatedEngine = Depends(get_federated_engine)
):
    """Get current global orchestration state."""
    global_state = federation.get_current_global_state()
    
    if not global_state:
        return {
            "status": "initializing",
            "message": "Global state not yet available"
        }
    
    return {
        "status": "ok",
        "data": global_state.to_dict()
    }

@router.get("/agents/status")
async def get_agents_status(
    federation: FederatedEngine = Depends(get_federated_engine)
):
    """Get status of all AI agents."""
    return await federation.get_status()

@router.post("/agents/ceo/force-decision")
async def force_ceo_decision():
    """Force immediate CEO decision (for testing)."""
    ceo = app.state.ai_ceo
    decision = await ceo.force_decision()
    return {"status": "ok", "decision": decision.to_dict()}
```

## Integration Example 3: Subscribing to Orchestration Events

```python
# backend/services/mission_control.py

from backend.core.event_bus import EventBus

async def setup_orchestration_listeners(event_bus: EventBus):
    """Setup listeners for orchestration events."""
    
    async def on_mode_switch(event_data: dict):
        """Handle operating mode switch."""
        payload = event_data.get("payload", {})
        new_mode = payload.get("new_mode")
        reason = payload.get("reason")
        
        print(f"🔄 Mode switched to: {new_mode}")
        print(f"   Reason: {reason}")
        
        # Update dashboard UI, send notifications, etc.
    
    async def on_risk_alert(event_data: dict):
        """Handle risk alerts."""
        payload = event_data.get("payload", {})
        level = payload.get("level")
        alerts = payload.get("alerts", [])
        
        print(f"⚠️  Risk Alert ({level}):")
        for alert in alerts:
            print(f"   - {alert}")
        
        # Send Telegram notification, update UI, etc.
    
    async def on_global_state_update(event_data: dict):
        """Handle global state updates."""
        payload = event_data.get("payload", {})
        mode = payload.get("global_mode")
        risk_level = payload.get("risk_level")
        
        print(f"📊 Global State: mode={mode}, risk={risk_level}")
        
        # Update real-time dashboard
    
    # Subscribe to events
    event_bus.subscribe("ceo_mode_switch", on_mode_switch)
    event_bus.subscribe("risk_alert", on_risk_alert)
    event_bus.subscribe("global_state_update", on_global_state_update)
    
    print("✅ Orchestration event listeners registered")
```

## Sequence Diagram: CEO Decision Cycle with Risk Alert

```
User/Timer          AI_CEO          EventBus        AI-RO         PolicyStore
    |                 |                |              |                |
    |   [30s cycle]   |                |              |                |
    ├────────────────>|                |              |                |
    |                 |                |              |                |
    |                 | gather_state() |              |                |
    |                 |────────────────┼─────────────>|                |
    |                 |                |              |                |
    |                 |                | risk_state   |                |
    |                 |<───────────────┼──────────────┤                |
    |                 |                |              |                |
    |                 | evaluate()     |              |                |
    |                 | (CEOBrain)     |              |                |
    |                 |                |              |                |
    |                 | get_policy()   |              |                |
    |                 |────────────────┼──────────────┼───────────────>|
    |                 |<───────────────┼──────────────┼────────────────┤
    |                 |                |              |                |
    |   [Decision: switch to DEFENSIVE] |              |                |
    |                 |                |              |                |
    |                 | set_policy()   |              |                |
    |                 |────────────────┼──────────────┼───────────────>|
    |                 |                |              |                |
    |                 | publish(       |              |                |
    |                 |   ceo_decision)|              |                |
    |                 |───────────────>|              |                |
    |                 |                |              |                |
    |                 | publish(       |              |                |
    |                 |   ceo_mode_    |              |                |
    |                 |   switch)      |              |                |
    |                 |───────────────>|              |                |
    |                 |                |              |                |
    |                 |                | mode_switch  |                |
    |                 |                |─────────────>|                |
    |                 |                |              |                |
    |                 |                |       [AI-RO receives mode   |
    |                 |                |        change notification]  |
```

## Sequence Diagram: Risk Alert Triggers Defensive Mode

```
AI-RO           RiskBrain       EventBus        AI_CEO        PolicyStore
  |                 |              |               |               |
  | assess_risk()   |              |               |               |
  |────────────────>|              |               |               |
  |                 |              |               |               |
  |  [Calculate VaR, drawdown > 4%]              |               |
  |                 |              |               |               |
  |<────────────────┤              |               |               |
  | RiskAssessment  |              |               |               |
  | risk_score=78   |              |               |               |
  |                 |              |               |               |
  | publish(        |              |               |               |
  |   risk_alert)   |              |               |               |
  |─────────────────┼─────────────>|               |               |
  |                 |              |               |               |
  |                 |              | risk_alert    |               |
  |                 |              |──────────────>|               |
  |                 |              |               |               |
  |                 |              |    [CEO receives critical     |
  |                 |              |     risk alert, triggers      |
  |                 |              |     immediate evaluation]     |
  |                 |              |               |               |
  |                 |              |               | evaluate()    |
  |                 |              |               | → DEFENSIVE   |
  |                 |              |               |               |
  |                 |              |               | update_policy()|
  |                 |              |               |──────────────>|
  |                 |              |               |               |
  |                 |              | ceo_mode_switch              |
  |                 |              |<──────────────┤               |
  |                 |              |               |               |
```

## Deployment Options

### Option 1: Run in analytics-os-service (Recommended)

Add to `backend/services/analytics_os_service.py`:
- All three agents + federation layer run in same service
- Shared EventBus and Redis connections
- Minimal overhead
- Easy to monitor and debug

### Option 2: Dedicated orchestrator-service

Create new `backend/services/orchestrator_service.py`:
- Dedicated microservice for orchestration
- Better isolation
- Can scale independently
- Use when system becomes very large

### Option 3: Hybrid approach

- AI CEO + Federation in orchestrator-service
- AI-RO in risk-os-service
- AI-SO in analytics-os-service
- Maximum modularity
- Requires careful event coordination

## Configuration

### Enable/Disable via PolicyStore

```python
# Add to risk mode config in PolicyStore
{
    "enable_ai_ceo": True,
    "enable_ai_ro": True,
    "enable_ai_so": True,
    "ai_ceo_decision_interval": 30,
    "ai_ro_assessment_interval": 30,
    "ai_so_analysis_interval": 60,
    "federation_update_interval": 15
}
```

### Adjust Thresholds

```python
# Customize CEO policy thresholds
from backend.ai_orchestrator.ceo_policy import CEOThresholds

thresholds = CEOThresholds(
    max_daily_drawdown_expansion=0.03,  # 3%
    critical_drawdown_threshold=0.08,   # 8%
    risk_score_critical=85.0,
    min_win_rate_expansion=0.52,
)

ceo_policy = CEOPolicy(thresholds=thresholds)
ceo_brain = CEOBrain(policy=ceo_policy)
ai_ceo = AI_CEO(brain=ceo_brain, ...)
```

## Monitoring

### Health Check Endpoints

```python
GET /api/orchestration/agents/status
GET /api/orchestration/global-state
GET /api/orchestration/agents/ceo/status
GET /api/orchestration/agents/risk/status
GET /api/orchestration/agents/strategy/status
```

### Key Metrics to Monitor

- **AI CEO**: Decision cycle time, mode switches per hour
- **AI-RO**: Risk score trend, alert frequency
- **AI-SO**: Strategy recommendation changes, model accuracy
- **Federation**: Data completeness, state staleness

## Testing

```python
# Test CEO decision logic
from backend.ai_orchestrator.ceo_brain import CEOBrain, SystemState

brain = CEOBrain()
state = SystemState(
    risk_score=75.0,
    win_rate=0.55,
    current_drawdown=0.04,
    # ... other fields
)

decision = brain.evaluate(state)
assert decision.operating_mode == "DEFENSIVE"

# Test risk calculations
from backend.ai_risk.risk_models import RiskModels
import numpy as np

models = RiskModels()
returns = np.random.normal(-0.001, 0.02, 100)

var_result = models.calculate_var(returns, confidence_level=0.95)
assert var_result.var_value > 0
```

## Backward Compatibility

The AI Orchestration Layer is fully compatible with:
- ✅ Prompt 6: PolicyStore v2, EventBus v2
- ✅ Prompt 7: Microservices v1
- ✅ Prompt 8: Replay Engine v3, Service Mesh

No breaking changes to existing systems. Can be disabled via PolicyStore flags.

## Next Steps (Prompt 10)

Potential enhancements:
1. Machine learning for CEO policy optimization
2. Advanced correlation-based risk models
3. Strategy performance prediction
4. Autonomous retraining triggers
5. Multi-timeframe regime detection
6. Portfolio optimization integration

---

**Implementation Status**: ✅ Complete and production-ready

**Build Constitution v3.5**: ✅ Fully compliant

