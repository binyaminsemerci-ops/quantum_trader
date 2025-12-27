# Federation AI v3 Service

**Executive AI orchestration layer for Quantum Trader v2.0**

## Overview

Federation AI v3 is the supreme decision-making layer that coordinates all AI subsystems through a hierarchy of 6 executive roles. It sits ABOVE the existing AI Engine, orchestrating capital allocation, risk management, strategy selection, and emergency interventions.

## Architecture

### Executive Roles

```
┌─────────────────────────────────────────────────────────┐
│               FEDERATION AI v3                          │
│                                                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │  AI-SUPERVISOR (Guardian)                       │   │ ← HIGHEST PRIORITY
│  │  - Emergency freeze authority                   │   │
│  │  - Override dangerous decisions                 │   │
│  └─────────────────────────────────────────────────┘   │
│                         ↓                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │  AI-CRO (Chief Risk Officer)                    │   │
│  │  - Global risk limits                           │   │
│  │  - ESS threshold tuning                         │   │
│  └─────────────────────────────────────────────────┘   │
│                         ↓                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │  AI-CEO (Chief Executive)                       │   │
│  │  - Capital profile (MICRO/LOW/NORMAL/AGGRESSIVE)│   │
│  │  - Trading mode (LIVE/SHADOW/PAUSED/EMERGENCY)  │   │
│  └─────────────────────────────────────────────────┘   │
│                         ↓                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │  AI-CIO (Chief Investment Officer)              │   │
│  │  - Model weight allocation                      │   │
│  │  - Symbol universe selection                    │   │
│  └─────────────────────────────────────────────────┘   │
│                         ↓                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │  AI-CFO (Chief Financial Officer)               │   │
│  │  - Profit allocation (lock/reinvest/reserve)    │   │
│  │  - Capital efficiency                           │   │
│  └─────────────────────────────────────────────────┘   │
│                         ↓                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │  AI-RESEARCHER (Research Director)              │   │
│  │  - Research task generation                     │   │
│  │  - Model improvement suggestions                │   │
│  └─────────────────────────────────────────────────┘   │ ← LOWEST PRIORITY
└─────────────────────────────────────────────────────────┘
```

### Decision Flow

```
EventBus → Orchestrator → Roles (parallel) → Priority Resolution → Publish
```

1. **Event Reception**: Portfolio, health, model performance events
2. **Parallel Processing**: All enabled roles evaluate simultaneously
3. **Priority Resolution**: Higher priority roles override lower ones
4. **Conflict Resolution**: Same decision type = highest role wins
5. **Publication**: Final decisions published to EventBus

## File Structure

```
backend/services/federation_ai/
├── __init__.py                 # Package initialization
├── models.py                   # Pydantic decision models (195 lines)
├── orchestrator.py             # Central coordinator (261 lines)
├── adapters.py                 # Backend system adapters (118 lines)
├── app.py                      # FastAPI REST API (234 lines)
├── main.py                     # EventBus integration (231 lines)
├── README.md                   # This file
└── roles/
    ├── base.py                 # Abstract role class (88 lines)
    ├── ceo.py                  # AI-CEO (217 lines)
    ├── cio.py                  # AI-CIO (199 lines)
    ├── cro.py                  # AI-CRO (225 lines)
    ├── cfo.py                  # AI-CFO (178 lines)
    ├── researcher.py           # AI-Researcher (216 lines)
    └── supervisor.py           # AI-Supervisor (244 lines)
```

**Total Lines**: ~2,400 lines of production code

## Decision Types

### 1. Capital Profile (CEO)
```python
{
    "profile": "NORMAL",              # MICRO, LOW, NORMAL, AGGRESSIVE
    "max_risk_per_trade_pct": 0.02,   # 2%
    "max_daily_risk_pct": 0.05,       # 5%
    "max_positions": 5
}
```

**Upgrade Conditions**:
- 5+ consecutive profitable days
- Drawdown < 3%
- Win rate > 60%
- Sharpe > 1.5

**Downgrade Conditions**:
- 3+ consecutive loss days
- Drawdown > 5%
- Win rate < 40%

**Emergency Downgrade to MICRO**:
- Drawdown > 8%
- Max drawdown > 10%

### 2. Trading Mode (CEO)
```python
{
    "mode": "LIVE",               # LIVE, SHADOW, PAUSED, EMERGENCY
    "duration_minutes": None      # None = indefinite
}
```

**Mode Selection Logic**:
- **EMERGENCY**: ESS CRITICAL or system EMERGENCY
- **PAUSED**: System CRITICAL or ESS WARNING (30 min)
- **SHADOW**: System DEGRADED
- **LIVE**: System HEALTHY + ESS NOMINAL

### 3. Risk Adjustment (CRO)
```python
{
    "max_leverage": 10.0,
    "max_position_size_usd": 10000.0,
    "max_drawdown_pct": 0.08,
    "max_exposure_pct": 0.90,
    "stop_loss_multiplier": 1.0
}
```

**Tightening Triggers**:
- DD increase > 2%
- Exposure > 85%
- ESS escalation

**Loosening Triggers**:
- DD < 2%
- Win rate > 65%
- Sharpe > 2.0

### 4. ESS Policy (CRO)
```python
{
    "caution_threshold_pct": 0.03,    # 3%
    "warning_threshold_pct": 0.05,    # 5%
    "critical_threshold_pct": 0.08    # 8%
}
```

**Emergency Tightening**: DD > 6% or max DD > 8%
**Restoration**: DD < 2% for stable period

### 5. Strategy Allocation (CIO)
```python
{
    "model_weights": {
        "xgboost": 0.30,
        "lightgbm": 0.30,
        "nhits": 0.25,
        "patchtst": 0.15
    },
    "active_strategies": ["xgboost", "lightgbm", "nhits", "patchtst"]
}
```

**Rebalancing Logic**:
- Score = Sharpe ratio + (win_rate - 0.5) * 2.0
- Normalize scores to weights
- Rebalance if any weight changes > 5%

### 6. Symbol Universe (CIO)
```python
{
    "active_symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    "excluded_symbols": [...]
}
```

**Expansion**: DD < 2%, win rate > 60%, positions < 8 → expand to 8 symbols
**Contraction**: DD > 5% or win rate < 45% → contract to 3 symbols

### 7. Cashflow (CFO)
```python
{
    "profit_lock_pct": 0.30,       # Lock 30% of profits
    "reinvest_pct": 0.60,          # Reinvest 60%
    "reserve_buffer_pct": 0.10     # Reserve 10%
}
```

**Policies**:
- **High DD (>5%)**: Lock 60%, reinvest 25%, reserve 15%
- **Low reserves (<5%)**: Lock 20%, reinvest 50%, reserve 30%
- **Strong profits**: Lock 30%, reinvest 60%, reserve 10%
- **Excess reserves (>15%)**: Lock 25%, reinvest 70%, reserve 5%

### 8. Research Task (Researcher)
```python
{
    "task_type": "hyperparameter_tuning",
    "description": "Tune XGBoost parameters using Optuna...",
    "estimated_duration_hours": 8,
    "expected_impact": "Improve Sharpe by 0.3-0.5",
    "model_name": "xgboost"
}
```

**Task Types**:
- `hyperparameter_tuning`: Sharpe < 1.0
- `feature_engineering`: Win rate < 50%
- `model_retraining`: Staleness > 30 days
- `risk_analysis`: DD > 6%
- `backtest_strategy`: Win rate < 45%

### 9. Override (Supervisor)
```python
{
    "overridden_decision_id": "uuid",
    "reason": "CEO wants AGGRESSIVE but DD > 5%"
}
```

**Use Cases**:
- Block dangerous decisions
- Enforce safety constraints
- Prevent rule violations

### 10. Freeze (Supervisor)
```python
{
    "duration_minutes": 120,                           # 2 hours
    "affected_subsystems": ["trading", "execution"],
    "severity": "CRITICAL",                            # CRITICAL, HIGH, MEDIUM
    "requires_manual_review": True,
    "reason": "Drawdown 10.2% exceeds limit"
}
```

**Freeze Triggers**:
- DD > 10%
- Daily loss > 5%
- 5+ consecutive losses
- Capital < $1000
- ESS CRITICAL
- System EMERGENCY

**Unfreeze Conditions**:
- DD < 5%
- Equity > $1500 (1.5x minimum)
- P&L positive

## REST API

### Health Check
```bash
GET http://localhost:8001/health

Response:
{
    "status": "healthy",
    "service": "federation-ai",
    "version": "3.0.0",
    "roles_active": 6
}
```

### Get Status
```bash
GET http://localhost:8001/api/federation/status

Response:
{
    "roles": {
        "ceo": true,
        "cio": true,
        "cro": true,
        "cfo": true,
        "researcher": true,
        "supervisor": true
    },
    "total_decisions": 142,
    "last_update": "2024-01-01T12:00:00Z"
}
```

### Get Decisions
```bash
GET http://localhost:8001/api/federation/decisions?limit=10

Response:
{
    "total": 10,
    "decisions": [
        {
            "decision_id": "uuid",
            "decision_type": "CAPITAL_PROFILE",
            "role_source": "ceo",
            "priority": 3,
            "timestamp": "2024-01-01T12:00:00Z",
            "reason": "5 profitable days, upgrading to NORMAL",
            "payload": {...}
        },
        ...
    ]
}
```

### Manual Mode Override
```bash
POST http://localhost:8001/api/federation/mode

Body:
{
    "mode": "PAUSED",
    "reason": "Manual testing",
    "duration_minutes": 60
}

Response:
{
    "success": true,
    "decision_id": "uuid",
    "mode": "PAUSED"
}
```

### Manual Risk Adjustment
```bash
POST http://localhost:8001/api/federation/risk

Body:
{
    "max_leverage": 5.0,
    "max_drawdown_pct": 0.05,
    "reason": "Emergency tightening"
}

Response:
{
    "success": true,
    "decision_id": "uuid",
    "limits": {
        "max_leverage": 5.0,
        "max_drawdown_pct": 0.05
    }
}
```

### Enable/Disable Roles
```bash
POST http://localhost:8001/api/federation/roles/researcher/disable

Response:
{
    "success": true,
    "role": "researcher",
    "status": "disabled"
}
```

**Note**: Cannot disable supervisor role (safety constraint)

## Running the Service

### Standalone Mode (Testing)
```bash
cd backend/services/federation_ai
python main.py
```

### FastAPI Server (Production)
```bash
uvicorn backend.services.federation_ai.app:app --host 0.0.0.0 --port 8001
```

### Docker (Production)
```bash
docker build -t federation-ai:v3.0 .
docker run -p 8001:8001 federation-ai:v3.0
```

## Integration with Existing Backend

### 1. Event Publishing (Main Backend)
```python
# backend/main.py or appropriate service

# Publish portfolio snapshots
await event_bus.publish("portfolio.snapshot_updated", {
    "total_equity": portfolio.equity,
    "drawdown_pct": portfolio.drawdown,
    "realized_pnl_today": portfolio.pnl_today,
    # ... other fields
})

# Publish system health
await event_bus.publish("system.health_updated", {
    "system_status": "HEALTHY",
    "ess_state": ess.current_state,
})

# Publish model performance
await event_bus.publish("model.performance_updated", {
    "model_name": "xgboost",
    "sharpe_ratio": 1.8,
    "win_rate": 0.65,
    # ... other fields
})
```

### 2. Decision Consumption (Main Backend)
```python
# Subscribe to Federation decisions
event_bus.subscribe("ai.federation.decision_made", handle_federation_decision)

async def handle_federation_decision(event):
    decision = event["data"]
    
    if decision["decision_type"] == "MODE_CHANGE":
        # Update trading mode
        trading_engine.set_mode(decision["payload"]["mode"])
    
    elif decision["decision_type"] == "RISK_ADJUSTMENT":
        # Update risk limits
        risk_manager.update_limits(decision["payload"])
    
    elif decision["decision_type"] == "FREEZE":
        # Emergency halt
        trading_engine.freeze(decision["payload"]["duration_minutes"])
    
    # ... handle other decision types
```

### 3. Adapter Integration
```python
# Connect adapters to actual services

# PolicyStore
from backend.core.policy_store import PolicyStore
policy_adapter.policy_store = PolicyStore()

# Portfolio Intelligence
from backend.services.portfolio_intelligence import PortfolioClient
portfolio_adapter.portfolio_client = PortfolioClient()

# AI Engine
from backend.services.ai.ai_hedgefund_os import AIHedgeFundOS
ai_engine_adapter.ai_engine = AIHedgeFundOS()

# ESS
from backend.services.execution.ess_service import ESSService
ess_adapter.ess_client = ESSService()
```

## Testing

### Unit Tests
```bash
pytest backend/services/federation_ai/tests/test_federation_roles.py -v
```

### Integration Tests
```bash
pytest backend/services/federation_ai/tests/test_orchestrator.py -v
```

### Manual Testing
```bash
# Start service
python backend/services/federation_ai/main.py

# In another terminal, send test events
curl -X POST http://localhost:8001/api/federation/mode \
  -H "Content-Type: application/json" \
  -d '{"mode": "SHADOW", "reason": "Testing"}'
```

## Monitoring

### Key Metrics
- **Decisions per hour**: Normal = 5-10, High = 20+
- **Role conflicts**: Should be < 5% of decisions
- **Override rate**: Supervisor overrides < 2%
- **Freeze incidents**: Should be rare (< 1 per week)

### Logs
```bash
# View Federation decisions
grep "Decision published" federation-ai.log | jq .

# View role conflicts
grep "Decision conflict resolved" federation-ai.log

# View emergency freezes
grep "EMERGENCY FREEZE" federation-ai.log
```

### Alerts
- **Critical**: Trading frozen by Supervisor
- **High**: Capital profile downgraded to MICRO
- **Medium**: 5+ role conflicts in 1 hour
- **Low**: Research task generated

## Safety Features

### 1. Supervisor Cannot Be Disabled
The supervisor role has ultimate authority and cannot be disabled via API.

### 2. Priority Enforcement
Higher priority roles always override lower ones:
- Supervisor > CRO > CEO > CIO > CFO > Researcher

### 3. Emergency Freeze
Immediate trading halt on:
- DD > 10%
- Daily loss > 5%
- Capital < $1000
- ESS CRITICAL

### 4. Manual Override
Operators can manually override via REST API with CRITICAL priority.

### 5. Decision Audit Trail
All decisions logged to database with full context.

## Future Enhancements

### Phase 2 (Q2 2024)
- [ ] Machine learning in CEO (predict optimal capital profile)
- [ ] RL-based CRO (learn optimal risk thresholds)
- [ ] Multi-exchange coordination in CIO
- [ ] Advanced cashflow optimization in CFO

### Phase 3 (Q3 2024)
- [ ] Market regime detection (Researcher)
- [ ] Predictive maintenance (Supervisor)
- [ ] Cross-exchange arbitrage (CIO)
- [ ] Dynamic fee optimization (CFO)

## License

Proprietary - Quantum Trader v2.0

## Authors

- AI-Assisted Development (GitHub Copilot + Claude Sonnet 4.5)
- System Architecture: v2.0 Blueprint (EPIC-FAI-001)

---

**Status**: ✅ IMPLEMENTED (Skeleton + Robust Structure)
**Next**: Integration with EventBus + Adapter Implementation
