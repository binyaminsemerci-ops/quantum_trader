# Federation AI v3 - Integration Guide

**For Backend Developers**

This guide shows how to integrate Federation AI v3 with the existing Quantum Trader backend.

---

## Step 1: Install Dependencies

Add to `backend/requirements.txt`:
```
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.5.0
structlog>=23.2.0
```

```bash
# In Docker container
pip install -r backend/requirements.txt
```

---

## Step 2: Publish Portfolio Snapshots

In `backend/services/portfolio_balancer.py` or `backend/services/portfolio_intelligence/portfolio_manager.py`:

```python
from backend.core.event_bus import event_bus

async def publish_portfolio_snapshot():
    """Publish portfolio snapshot every 5 seconds"""
    
    # Get current portfolio state
    portfolio = await get_portfolio_state()
    
    # Publish to EventBus
    await event_bus.publish("portfolio.snapshot_updated", {
        "total_equity": portfolio.equity,
        "drawdown_pct": portfolio.drawdown_pct,
        "max_drawdown_pct": portfolio.max_drawdown_pct,
        "realized_pnl_today": portfolio.realized_pnl_today,
        "unrealized_pnl": portfolio.unrealized_pnl,
        "num_positions": len(portfolio.positions),
        "total_exposure_usd": sum(p.size_usd for p in portfolio.positions),
        "win_rate_today": portfolio.calculate_win_rate(),
        "sharpe_ratio_7d": portfolio.calculate_sharpe(days=7),
    })

# Add to background task loop
asyncio.create_task(periodic_portfolio_snapshot())
```

---

## Step 3: Publish System Health

In `backend/main.py` or health check service:

```python
from backend.core.event_bus import event_bus
from backend.services.execution.ess_service import ess_service

async def publish_system_health():
    """Publish system health every 30 seconds"""
    
    # Get system status
    system_status = get_system_status()  # "OPTIMAL", "HEALTHY", "DEGRADED", "CRITICAL", "EMERGENCY"
    
    # Get ESS state
    ess_state = ess_service.get_current_state()  # "NOMINAL", "CAUTION", "WARNING", "CRITICAL"
    
    # Publish to EventBus
    await event_bus.publish("system.health_updated", {
        "system_status": system_status,
        "ess_state": ess_state,
    })

# Add to background task loop
asyncio.create_task(periodic_health_check())
```

---

## Step 4: Publish Model Performance

In `backend/services/ai/ai_hedgefund_os.py` or after each trade:

```python
from backend.core.event_bus import event_bus

async def publish_model_performance(model_name: str):
    """Publish model performance after each trade or daily"""
    
    # Get model metrics
    metrics = await get_model_metrics(model_name)
    
    # Publish to EventBus
    await event_bus.publish("model.performance_updated", {
        "model_name": model_name,
        "sharpe_ratio": metrics.sharpe_ratio,
        "win_rate": metrics.win_rate,
        "profit_factor": metrics.profit_factor,
        "total_trades": metrics.total_trades,
        "days_since_training": metrics.days_since_training,
    })

# Call after each trade closes
await publish_model_performance("xgboost")
```

---

## Step 5: Subscribe to Federation Decisions

In `backend/main.py` startup:

```python
from backend.core.event_bus import event_bus
from backend.core.policy_store import policy_store
from backend.services.execution.trading_engine import trading_engine
from backend.services.execution.ess_service import ess_service

async def handle_federation_decision(event: dict):
    """Handle Federation AI decisions"""
    
    decision = event["data"]
    decision_type = decision["decision_type"]
    payload = decision["payload"]
    
    if decision_type == "MODE_CHANGE":
        # Update trading mode
        mode = payload["mode"]
        logger.info(f"Federation: Trading mode changed to {mode}")
        
        if mode == "EMERGENCY":
            await trading_engine.emergency_stop()
        elif mode == "PAUSED":
            await trading_engine.pause(duration_minutes=payload.get("duration_minutes"))
        elif mode == "SHADOW":
            await trading_engine.set_shadow_mode()
        elif mode == "LIVE":
            await trading_engine.set_live_mode()
    
    elif decision_type == "CAPITAL_PROFILE":
        # Update capital profile in PolicyStore
        profile = payload["profile"]
        await policy_store.set("capital_profile", profile)
        await policy_store.set("max_risk_per_trade_pct", payload["max_risk_per_trade_pct"])
        await policy_store.set("max_daily_risk_pct", payload["max_daily_risk_pct"])
        await policy_store.set("max_positions", payload["max_positions"])
        logger.info(f"Federation: Capital profile changed to {profile}")
    
    elif decision_type == "RISK_ADJUSTMENT":
        # Update risk limits
        await policy_store.set("max_leverage", payload.get("max_leverage"))
        await policy_store.set("max_position_size_usd", payload.get("max_position_size_usd"))
        await policy_store.set("max_drawdown_pct", payload.get("max_drawdown_pct"))
        await policy_store.set("max_exposure_pct", payload.get("max_exposure_pct"))
        await policy_store.set("stop_loss_multiplier", payload.get("stop_loss_multiplier"))
        logger.info(f"Federation: Risk limits updated")
    
    elif decision_type == "ESS_POLICY":
        # Update ESS thresholds
        await ess_service.update_thresholds(
            caution=payload["caution_threshold_pct"],
            warning=payload["warning_threshold_pct"],
            critical=payload["critical_threshold_pct"]
        )
        logger.info(f"Federation: ESS thresholds updated")
    
    elif decision_type == "STRATEGY_ALLOCATION":
        # Update model weights in AI Engine
        weights = payload["model_weights"]
        await ai_engine.set_model_weights(weights)
        logger.info(f"Federation: Strategy allocation updated: {weights}")
    
    elif decision_type == "SYMBOL_UNIVERSE":
        # Update active symbols
        active_symbols = payload["active_symbols"]
        await policy_store.set("active_symbols", active_symbols)
        logger.info(f"Federation: Active symbols: {active_symbols}")
    
    elif decision_type == "CASHFLOW":
        # Update cashflow policy
        await policy_store.set("profit_lock_pct", payload["profit_lock_pct"])
        await policy_store.set("reinvest_pct", payload["reinvest_pct"])
        await policy_store.set("reserve_buffer_pct", payload["reserve_buffer_pct"])
        logger.info(f"Federation: Cashflow policy updated")
    
    elif decision_type == "FREEZE":
        # Emergency freeze
        duration = payload.get("duration_minutes", 0)
        severity = payload.get("severity")
        reason = payload.get("reason")
        
        logger.critical(f"Federation FREEZE: {reason} (severity: {severity})")
        
        if duration == 0:
            # Indefinite freeze
            await trading_engine.freeze_indefinite(reason=reason)
        else:
            await trading_engine.freeze(duration_minutes=duration, reason=reason)
    
    elif decision_type == "RESEARCH_TASK":
        # Log research task (could trigger AutoML pipeline)
        logger.info(
            f"Federation Research Task: {payload['task_type']}",
            description=payload["description"],
            impact=payload.get("expected_impact")
        )
    
    elif decision_type == "OVERRIDE":
        # Log override (informational)
        logger.warning(
            f"Federation Override: {payload['reason']}",
            overridden_decision=payload["overridden_decision_id"]
        )

# Subscribe on startup
event_bus.subscribe("ai.federation.decision_made", handle_federation_decision)
```

---

## Step 6: Wire Up Adapters

In `backend/services/federation_ai/main.py`:

```python
from backend.core.policy_store import PolicyStore
from backend.services.portfolio_intelligence.portfolio_manager import PortfolioManager
from backend.services.ai.ai_hedgefund_os import AIHedgeFundOS
from backend.services.execution.ess_service import ESSService

# In FederationAIService.__init__()
self.policy_store.policy_store = PolicyStore()
self.portfolio.portfolio_client = PortfolioManager()
self.ai_engine.ai_engine = AIHedgeFundOS()
self.ess.ess_client = ESSService()
```

---

## Step 7: Add to Docker Compose

In `docker-compose.yml`:

```yaml
services:
  backend:
    # ... existing config
  
  federation-ai:
    build:
      context: .
      dockerfile: backend/services/federation_ai/Dockerfile
    container_name: quantum_federation_ai
    ports:
      - "8001:8001"
    environment:
      - REDIS_URL=redis://redis:6379
      - DB_URL=postgresql://user:pass@postgres:5432/quantum_trader
      - LOG_LEVEL=INFO
    depends_on:
      - redis
      - postgres
    networks:
      - quantum_network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

Create `backend/services/federation_ai/Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Federation AI code
COPY backend/services/federation_ai ./backend/services/federation_ai
COPY backend/core ./backend/core

# Expose port
EXPOSE 8001

# Run FastAPI app
CMD ["uvicorn", "backend.services.federation_ai.app:app", "--host", "0.0.0.0", "--port", "8001"]
```

---

## Step 8: Dashboard Integration

Add Federation AI widget to dashboard:

```typescript
// frontend/src/components/FederationAIWidget.tsx

interface FederationStatus {
  roles: Record<string, boolean>;
  total_decisions: number;
  last_update: string | null;
}

export function FederationAIWidget() {
  const [status, setStatus] = useState<FederationStatus | null>(null);
  
  useEffect(() => {
    const fetchStatus = async () => {
      const response = await fetch('http://localhost:8001/api/federation/status');
      const data = await response.json();
      setStatus(data);
    };
    
    fetchStatus();
    const interval = setInterval(fetchStatus, 10000); // Every 10 seconds
    
    return () => clearInterval(interval);
  }, []);
  
  if (!status) return <div>Loading...</div>;
  
  return (
    <Card>
      <CardHeader>
        <CardTitle>Federation AI v3</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          <div className="flex justify-between">
            <span>Active Roles:</span>
            <span className="font-mono">{Object.values(status.roles).filter(Boolean).length}/6</span>
          </div>
          <div className="flex justify-between">
            <span>Total Decisions:</span>
            <span className="font-mono">{status.total_decisions}</span>
          </div>
          <div className="flex justify-between">
            <span>Last Update:</span>
            <span className="font-mono text-xs">{status.last_update ? new Date(status.last_update).toLocaleTimeString() : 'N/A'}</span>
          </div>
          
          <Separator className="my-4" />
          
          <div className="space-y-1">
            {Object.entries(status.roles).map(([role, enabled]) => (
              <div key={role} className="flex items-center justify-between">
                <span className="text-sm capitalize">{role}</span>
                <Badge variant={enabled ? "success" : "secondary"}>
                  {enabled ? "Enabled" : "Disabled"}
                </Badge>
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
```

---

## Step 9: Testing

### Test Event Publishing
```bash
# Start backend
docker-compose up backend

# Check logs for portfolio.snapshot_updated events
docker logs quantum_backend | grep "portfolio.snapshot_updated"
```

### Test Federation AI
```bash
# Start Federation AI
docker-compose up federation-ai

# Check health
curl http://localhost:8001/health

# Check status
curl http://localhost:8001/api/federation/status

# Trigger manual mode change
curl -X POST http://localhost:8001/api/federation/mode \
  -H "Content-Type: application/json" \
  -d '{"mode": "SHADOW", "reason": "Testing integration", "duration_minutes": 5}'

# Check decisions
curl http://localhost:8001/api/federation/decisions?limit=10
```

### Test Decision Handling
```bash
# Check backend logs for Federation decision handling
docker logs quantum_backend | grep "Federation:"
```

---

## Step 10: Monitoring

### Grafana Dashboard

Add panels for:
- Federation decisions per hour (gauge)
- Role status (table)
- Capital profile over time (line chart)
- Trading mode over time (state timeline)
- Emergency freezes (counter)

### Alerts

Configure alerts for:
- **Critical**: Trading frozen by Supervisor
- **High**: Capital profile downgraded to MICRO
- **Medium**: 5+ role conflicts in 1 hour
- **Low**: Federation AI service unhealthy

---

## Troubleshooting

### Issue: No decisions being made
**Check**:
1. Are portfolio snapshots being published? (`grep "portfolio.snapshot_updated" backend.log`)
2. Is Federation AI receiving events? (`grep "Portfolio update received" federation-ai.log`)
3. Are roles enabled? (`curl http://localhost:8001/api/federation/status`)

### Issue: Decisions not being applied
**Check**:
1. Is backend subscribed to `ai.federation.decision_made`? (check `event_bus.subscribe` call)
2. Is `handle_federation_decision()` being called? (`grep "Federation:" backend.log`)
3. Are decision handlers implemented for all types?

### Issue: Federation AI service crashes
**Check**:
1. Are all dependencies installed? (`pip list | grep -E "fastapi|pydantic|structlog"`)
2. Are import paths correct? (relative imports use `.` prefix)
3. Is EventBus connection working? (check Redis connectivity)

---

## Next Steps

1. ✅ Complete Steps 1-10 above
2. ✅ Deploy to staging environment
3. ✅ Run integration tests for 24 hours
4. ✅ Monitor decision accuracy and system stability
5. ✅ Deploy to production

---

**Questions?** Check the [Federation AI README](./backend/services/federation_ai/README.md) for detailed documentation.
