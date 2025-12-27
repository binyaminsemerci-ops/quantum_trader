# 📊 SPRINT 4 - PART 1: BACKEND API ANALYSIS

## Existing API Endpoints

### **Portfolio Intelligence Service** (`:8004`)
**Prefix**: `/api/portfolio`

- `GET /snapshot` → PortfolioSnapshot (equity, cash, positions, total_pnl, daily_pnl)
- `GET /pnl` → PnLBreakdown (realized, unrealized, daily, weekly, monthly)
- `GET /exposure` → ExposureBreakdown (total, long, short, net by symbol/sector)
- `GET /drawdown` → DrawdownMetrics (daily_dd%, weekly_dd%, max_dd%)
- `GET /health` → ServiceHealth

**Available Data**:
- ✅ Current equity, cash, margin
- ✅ Open positions (symbol, side, size, entry_price, current_price, unrealized_pnl)
- ✅ PnL breakdown (realized/unrealized)
- ✅ Drawdown metrics (daily%, max%)
- ✅ Exposure analysis

### **AI Engine Service** (`:8001`)
**Prefix**: `/api/ai`

- `POST /signal` → SignalResponse (symbol, direction, confidence, strategy, sizing)
- `GET /metrics/ensemble` → EnsembleMetrics (model performance, voting results)
- `GET /metrics/meta_strategy` → MetaStrategyMetrics (active strategy, performance)
- `GET /metrics/rl_sizing` → RLSizingMetrics (sizing decisions, risk-adjusted positions)
- `GET /health` → ServiceHealth

**Available Data**:
- ✅ Latest signals (symbol, direction, confidence)
- ✅ Active strategy (ensemble, meta, RL)
- ✅ Model performance metrics
- ⚠️ **Missing**: Historical signals list (last 10-20 signals)

### **Execution Service** (`:8002`)
**Prefix**: `/api/execution`

- `POST /order` → OrderResponse (place manual order)
- `GET /positions` → PositionListResponse (active positions)
- `GET /trades` → TradeListResponse (recent trades, filter by status/symbol)
- `GET /metrics` → ExecutionMetrics (orders_placed, fills, rejections)
- `GET /health` → ServiceHealth

**Available Data**:
- ✅ Active positions (symbol, side, quantity, entry, current_price, pnl)
- ✅ Recent trades (last N trades, filter by status)
- ✅ Execution metrics (order stats)
- ⚠️ **Missing**: Open orders list (pending orders)

### **Risk & Safety Service** (`:8003`)
**Prefix**: `/api/risk`

- `GET /ess/status` → ESSStatus (state: ARMED/TRIPPED/COOLING, reason, timestamp)
- `GET /policies` → PolicyList (active policies, limits)
- `POST /ess/reset` → Manual ESS reset
- `GET /health` → ServiceHealth

**Available Data**:
- ✅ ESS state (ARMED/TRIPPED)
- ✅ Risk policies (max_position, daily_loss_limit)
- ⚠️ **Missing**: Real-time risk metrics (current exposure%, margin used%)

### **Monitoring & Health Service** (`:8080`)
**Prefix**: `/api/health`

- `GET /services` → ServiceHealthList (status of all 6 microservices)
- `GET /alerts` → AlertList (recent alerts, filter by severity)
- `GET /metrics` → SystemMetrics (CPU, memory, disk, network)
- `GET /health` → ServiceHealth

**Available Data**:
- ✅ System-wide health (all services OK/DEGRADED/DOWN)
- ✅ Recent alerts (ESS trips, service failures)
- ✅ System metrics

## Data Gaps for Dashboard

**Missing/Needed**:
1. **Signals History**: Last 10-20 AI signals (not just metrics) - needs endpoint in ai-engine
2. **Open Orders**: Pending orders waiting for fill - needs endpoint in execution
3. **Real-time Risk**: Current exposure%, margin%, limits% - needs aggregation
4. **Event Stream**: WebSocket for live updates (position, PnL, ESS state changes)

## Dashboard API Requirements

**REST Endpoint**: `GET /api/dashboard/snapshot`
- Aggregates: portfolio snapshot + positions + latest signals + ESS state + system health
- Single request for initial dashboard load

**WebSocket Endpoint**: `WS /ws/dashboard`
- Events: `position_updated`, `pnl_updated`, `signal_generated`, `ess_state_changed`, `health_alert`
- Real-time updates to dashboard panels

---

**Status**: ✅ Analysis Complete  
**Next**: Design Dashboard API Contract (Part 2)
