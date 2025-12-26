# Quantum Fund Dashboard - Service Integration Plan

## Current Dashboard (quantumfond.com)
- ✅ Portfolio overview (mock data)
- ✅ AI insights (mock data)
- ✅ Real-time metrics (mock data)
- ✅ WebSocket live stream

## Services Ready for Integration

### 🎯 High Priority (Critical for Hedge Fund)

#### 1. **Portfolio Intelligence** (quantum_portfolio_intelligence)
- **Port**: 8004
- **Status**: ✅ Healthy
- **Data**: Real portfolio positions, P&L, allocations
- **Integration**:
  ```python
  GET http://localhost:8004/portfolio/summary
  GET http://localhost:8004/portfolio/positions
  GET http://localhost:8004/portfolio/performance
  ```
- **Dashboard Use**: Replace mock portfolio data with real positions

#### 2. **Trading Bot** (quantum_trading_bot)
- **Port**: 8003
- **Status**: ✅ Healthy
- **Data**: Active trades, signals, execution status
- **Integration**:
  ```python
  GET http://localhost:8003/trades/active
  GET http://localhost:8003/trades/history
  GET http://localhost:8003/signals/latest
  ```
- **Dashboard Use**: Show live trading activity, win rate, trade volume

#### 3. **AI Engine** (quantum_ai_engine)
- **Port**: 8001
- **Status**: ✅ Healthy
- **Data**: Model predictions, confidence scores, market forecasts
- **Integration**:
  ```python
  GET http://localhost:8001/predictions/latest
  GET http://localhost:8001/models/performance
  GET http://localhost:8001/confidence/scores
  ```
- **Dashboard Use**: Replace mock AI insights with real model predictions

#### 4. **Risk Brain** (quantum_risk_brain)
- **Port**: 8012
- **Status**: ✅ Healthy
- **Data**: Risk metrics, VaR, max drawdown, exposure
- **Integration**:
  ```python
  GET http://localhost:8012/risk/metrics
  GET http://localhost:8012/risk/var
  GET http://localhost:8012/risk/exposure
  ```
- **Dashboard Use**: Real-time risk monitoring section

#### 5. **Position Monitor** (quantum_position_monitor)
- **Port**: Internal
- **Status**: ✅ Healthy
- **Data**: Position tracking, entry/exit points
- **Integration**: Via quantum_backend
- **Dashboard Use**: Position details, stop-loss, take-profit levels

#### 6. **Trade Journal** (quantum_trade_journal)
- **Port**: Internal
- **Status**: ✅ Healthy
- **Data**: Historical trades, performance analysis
- **Integration**: Via quantum_backend
- **Dashboard Use**: Trade history table, performance analytics

### 📊 Medium Priority (Enhanced Metrics)

#### 7. **Strategy Brain** (quantum_strategy_brain)
- **Port**: 8011
- **Status**: ✅ Healthy
- **Data**: Strategy performance, backtests
- **Dashboard Use**: Strategy comparison, optimization metrics

#### 8. **CEO Brain** (quantum_ceo_brain)
- **Port**: 8010
- **Status**: ✅ Healthy
- **Data**: High-level decisions, capital allocation
- **Dashboard Use**: Executive dashboard view

#### 9. **Model Supervisor** (quantum_model_supervisor)
- **Port**: 8007
- **Status**: ✅ Healthy
- **Data**: Model health, drift detection, retraining status
- **Dashboard Use**: Model monitoring section

#### 10. **CLM - Continuous Learning** (quantum_clm)
- **Port**: Internal
- **Status**: ✅ Running
- **Data**: Learning progress, adaptation metrics
- **Dashboard Use**: AI adaptation and learning curves

#### 11. **RL Optimizer** (quantum_rl_optimizer)
- **Port**: Internal
- **Status**: ✅ Healthy
- **Data**: Reinforcement learning metrics, policy updates
- **Dashboard Use**: RL performance charts

#### 12. **Universe OS** (quantum_universe_os)
- **Port**: 8006
- **Status**: ✅ Healthy
- **Data**: Market universe, asset selection
- **Dashboard Use**: Asset universe overview

### 🔧 Low Priority (Technical/Admin)

#### 13. **Governance Dashboard** (quantum_governance_dashboard)
- **Port**: 8501
- **Status**: ✅ Healthy
- **Data**: Compliance, audit logs
- **Dashboard Use**: Compliance section (separate tab)

#### 14. **Grafana** (quantum_grafana)
- **Port**: 3001
- **Status**: ✅ Healthy
- **Data**: System metrics, infrastructure monitoring
- **Dashboard Use**: Embed Grafana panels for system health

#### 15. **Prometheus** (quantum_prometheus)
- **Port**: 9090
- **Status**: ✅ Healthy
- **Data**: Time-series metrics
- **Dashboard Use**: Backend metrics source

## Existing Dashboards to Integrate

### RL Dashboard (quantum_rl_dashboard)
- **Port**: 8025 (conflict with new dashboard!)
- **Status**: ✅ Running
- **Solution**: Move to different port or integrate into main dashboard

### Old Dashboard (quantum_dashboard)
- **Port**: 8080
- **Status**: ✅ Running
- **Solution**: Deprecate or merge features into quantumfond.com

## Integration Architecture

```
quantumfond.com Dashboard (Port 8025)
    ↓
    ├─→ Portfolio Intelligence (8004) → Real positions
    ├─→ Trading Bot (8003)           → Active trades
    ├─→ AI Engine (8001)             → Predictions
    ├─→ Risk Brain (8012)            → Risk metrics
    ├─→ Strategy Brain (8011)        → Strategy performance
    ├─→ CEO Brain (8010)             → Executive decisions
    ├─→ Model Supervisor (8007)      → Model health
    ├─→ Universe OS (8006)           → Market data
    └─→ Main Backend (8000)          → Orchestration
```

## Implementation Phases

### Phase 1: Core Financial Data (Week 1)
- [ ] Integrate Portfolio Intelligence API
- [ ] Connect Trading Bot for live trades
- [ ] Replace mock portfolio with real data
- [ ] Add real-time P&L tracking

### Phase 2: AI & Risk (Week 2)
- [ ] Integrate AI Engine predictions
- [ ] Add Risk Brain metrics
- [ ] Real-time model performance
- [ ] Risk alerts and monitoring

### Phase 3: Strategy & Performance (Week 3)
- [ ] Strategy Brain integration
- [ ] Trade Journal historical data
- [ ] Performance analytics
- [ ] Backtesting results

### Phase 4: Advanced Features (Week 4)
- [ ] CEO Brain executive view
- [ ] Model Supervisor monitoring
- [ ] CLM learning metrics
- [ ] RL Optimizer performance
- [ ] Grafana panel embedding

### Phase 5: Governance & Compliance (Week 5)
- [ ] Governance Dashboard integration
- [ ] Audit log viewer
- [ ] Compliance reporting
- [ ] Alert management

## Current Issues to Resolve

### ⚠️ Port Conflicts
1. **RL Dashboard**: Port 8025 conflicts with new dashboard backend
   - **Solution**: Move RL Dashboard to port 8026
   ```yaml
   quantum_rl_dashboard:
     ports:
       - "8026:8025"  # Changed from 8025:8025
   ```

2. **Old Dashboard**: Port 8080 
   - **Solution**: Deprecate or move to 8081

### ⚠️ Unhealthy Services
1. **quantum_dashboard_frontend**: Unhealthy
   - Check: `docker logs quantum_dashboard_frontend`
   - Likely: CORS or API connection issue

2. **quantum_nginx**: Exited
   - Already resolved - using system nginx instead

## API Aggregation Layer

Create unified API in dashboard backend:

```python
# dashboard_v4/backend/integrations/quantum_services.py

class QuantumServicesClient:
    """Aggregates data from all Quantum Trader services"""
    
    SERVICES = {
        'portfolio': 'http://localhost:8004',
        'trading': 'http://localhost:8003',
        'ai_engine': 'http://localhost:8001',
        'risk': 'http://localhost:8012',
        'strategy': 'http://localhost:8011',
        'ceo': 'http://localhost:8010',
        'model_supervisor': 'http://localhost:8007',
        'universe': 'http://localhost:8006',
        'backend': 'http://localhost:8000'
    }
    
    async def get_portfolio_summary(self):
        """Get real portfolio data"""
        return await self._get('portfolio', '/portfolio/summary')
    
    async def get_live_trades(self):
        """Get active trades"""
        return await self._get('trading', '/trades/active')
    
    async def get_ai_predictions(self):
        """Get AI model predictions"""
        return await self._get('ai_engine', '/predictions/latest')
    
    async def get_risk_metrics(self):
        """Get risk metrics"""
        return await self._get('risk', '/risk/metrics')
```

## Frontend Components to Update

### 1. Portfolio Section
```typescript
// Replace mock data with real API
const fetchPortfolio = async () => {
  const response = await fetch(`${API_URL}/integrations/portfolio`);
  return response.json();
};
```

### 2. Trading Activity
```typescript
// Show live trades from quantum_trading_bot
const fetchLiveTrades = async () => {
  const response = await fetch(`${API_URL}/integrations/trades/active`);
  return response.json();
};
```

### 3. AI Insights
```typescript
// Real predictions from quantum_ai_engine
const fetchAIInsights = async () => {
  const response = await fetch(`${API_URL}/integrations/ai/predictions`);
  return response.json();
};
```

## Benefits of Integration

1. **Real Data**: No more mock data - all metrics from live system
2. **Single View**: One dashboard to see everything
3. **Real-time Updates**: Live trading, positions, and risk
4. **Historical Analysis**: Access to trade journal and performance
5. **AI Transparency**: See model predictions and confidence
6. **Risk Management**: Live risk monitoring and alerts

## Next Steps

1. **Immediate**: Fix quantum_rl_dashboard port conflict
2. **This Week**: Integrate Portfolio Intelligence and Trading Bot
3. **Next Week**: Add AI Engine and Risk Brain
4. **Following**: Strategy, performance, and advanced features

## Documentation Updates Needed

- [ ] API documentation for each service endpoint
- [ ] Integration guide for developers
- [ ] Data flow diagrams
- [ ] Error handling and fallback strategies
- [ ] Testing plan for each integration

---

**Last Updated**: December 26, 2025
**Priority**: High - Transform mock dashboard into real production system
