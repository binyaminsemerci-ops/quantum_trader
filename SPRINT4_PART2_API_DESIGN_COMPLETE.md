# SPRINT 4 PART 2: API DESIGN - COMPLETE

**Status**: ✅ **COMPLETE**  
**Sprint**: 4 (Dashboard & Frontend)  
**Part**: 2 (Dashboard API Design)  
**Date**: 2025-01-27

---

## 📋 Objectives

Design clean API contract for dashboard:
1. REST endpoint for initial snapshot load
2. WebSocket endpoint for real-time updates
3. Data models for consistent JSON serialization
4. Event types for all real-time scenarios

---

## 📁 Files Created

### 1. `backend/api/dashboard/models.py` (350+ lines)

**Data Structures**:
- **Enums**: `ESSState`, `ServiceStatus`, `SignalDirection`, `PositionSide`, `EventType`
- **DashboardPosition**: symbol, side, size, entry/current price, PnL (absolute + %)
- **DashboardSignal**: timestamp, symbol, direction, confidence, strategy
- **DashboardPortfolio**: equity, cash, margin, PnL breakdown (daily/weekly/monthly), position count
- **DashboardRisk**: ESS state/reason, drawdown (daily/weekly/max), exposure (long/short/net), risk limit %
- **ServiceHealthInfo**: name, status, latency, last check
- **DashboardSystemHealth**: overall status, services list, alerts count
- **DashboardSnapshot**: Complete aggregation for REST endpoint (portfolio + positions + signals + risk + system)
- **DashboardEvent**: Real-time event structure for WebSocket (type + timestamp + payload)

**Helper Functions**:
- `create_position_updated_event()` - Position size/PnL changed
- `create_pnl_updated_event()` - Portfolio PnL changed
- `create_signal_generated_event()` - New AI signal
- `create_ess_state_changed_event()` - ESS state changed
- `create_health_alert_event()` - Service health alert

**Features**:
- All models have `.to_dict()` for JSON serialization
- Float values rounded to 2-4 decimals
- Comprehensive docstrings

---

### 2. `backend/api/dashboard/routes.py` (330+ lines)

**REST Endpoint**:
```
GET /api/dashboard/snapshot
```

**Aggregation Logic**:
1. **Portfolio Data**: Calls `/api/portfolio/snapshot` + `/api/portfolio/pnl`
2. **Positions**: Calls `/api/execution/positions`
3. **Signals**: Calls `/api/ai/signals/recent` (TODO: endpoint doesn't exist yet)
4. **Risk**: Calls `/api/risk/ess/status` + `/api/portfolio/drawdown` + `/api/portfolio/exposure`
5. **System Health**: Calls `/api/health/services` + `/api/health/alerts`

**Features**:
- Async parallel aggregation with `asyncio.gather()`
- Graceful degradation: returns default values if service down
- Error handling with exception catching
- Timeout protection (5 seconds per service)
- Health endpoint: `GET /api/dashboard/health`

**Service URLs** (from config):
- Portfolio: `http://portfolio-intelligence-service:8004`
- AI Engine: `http://ai-engine-service:8001`
- Execution: `http://execution-service:8002`
- Risk: `http://risk-safety-service:8003`
- Monitoring: `http://monitoring-health-service:8080`

**Response Example**:
```json
{
  "timestamp": "2025-01-27T10:00:00Z",
  "portfolio": {
    "equity": 100000,
    "cash": 50000,
    "total_pnl": 5000,
    "daily_pnl": 150.25,
    "daily_pnl_pct": 0.15,
    ...
  },
  "positions": [
    {
      "symbol": "BTCUSDT",
      "side": "LONG",
      "size": 0.5,
      "entry_price": 95000,
      "current_price": 96000,
      "unrealized_pnl": 500,
      "unrealized_pnl_pct": 1.05
    }
  ],
  "signals": [...],
  "risk": {...},
  "system": {...}
}
```

---

### 3. `backend/api/dashboard/websocket.py` (300+ lines)

**WebSocket Endpoint**:
```
WS /ws/dashboard
```

**Protocol**:
1. Client connects
2. Server sends `connected` confirmation
3. Server broadcasts events as they occur
4. Heartbeat every 60 seconds
5. Client can send `ping` → receives `pong`

**Event Types** (7):
- `position_updated` - Position size/PnL changed
- `pnl_updated` - Portfolio PnL changed
- `signal_generated` - New AI signal
- `ess_state_changed` - ESS tripped/armed/cooling
- `health_alert` - Service degraded/down
- `trade_executed` - Trade filled
- `order_placed` - New order submitted

**Features**:
- **ConnectionManager**: Handles multiple client connections
- **Broadcast**: Send events to all connected clients
- **Auto-cleanup**: Remove disconnected clients
- **EventBus Integration** (TODO): Subscribe to internal events
- **Heartbeat**: Keep-alive mechanism
- **Ping/Pong**: Client can test connection

**Event Message Example**:
```json
{
  "type": "position_updated",
  "timestamp": "2025-01-27T10:30:00Z",
  "payload": {
    "symbol": "BTCUSDT",
    "side": "LONG",
    "size": 0.5,
    "unrealized_pnl": 125.50
  }
}
```

**TODO**:
- Integrate with real EventBus (`infrastructure/event_bus.py`)
- Currently uses mock event generation for testing

---

## 🔌 API Contract Summary

### REST Snapshot (Initial Load)

**Endpoint**: `GET /api/dashboard/snapshot`

**Purpose**: Single request to load complete dashboard state

**Response Time**: 500ms - 2s (parallel aggregation)

**Data Included**:
- Portfolio: equity, cash, margin, PnL (total, daily, weekly, monthly, realized, unrealized)
- Positions: all open positions with live PnL
- Signals: last 10-20 AI signals
- Risk: ESS state, drawdown %, exposure, risk limit %
- System: service health, alerts

---

### WebSocket (Real-time Updates)

**Endpoint**: `WS /ws/dashboard`

**Purpose**: Push updates to client without polling

**Frequency**: 
- Position updates: ~1-10/min (trade-dependent)
- PnL updates: ~1/min (price updates)
- Signal generation: ~2-5/min (market-dependent)
- ESS state: rare (only on trip/cooldown)
- Health alerts: rare (only on degradation)

**Benefits**:
- No polling overhead
- Instant updates (< 100ms latency)
- Efficient bandwidth usage

---

## 🧪 Testing Plan

### Manual Testing (Part 3)

1. **Snapshot Endpoint**:
   ```bash
   curl http://localhost:8000/api/dashboard/snapshot
   ```
   - Verify JSON response structure
   - Check all fields populated
   - Test service down scenario (mock failure)

2. **WebSocket**:
   ```bash
   websocat ws://localhost:8000/ws/dashboard
   ```
   - Verify connection confirmation
   - Check heartbeat messages
   - Send `ping`, expect `pong`
   - Trigger internal event, verify broadcast

### Unit Tests (Part 6)

- `tests/unit/test_dashboard_api_sprint4.py`:
  - Test snapshot aggregation with mocked services
  - Test WebSocket connection/disconnection
  - Test event broadcasting to multiple clients
  - Test graceful degradation on service failure

---

## 📊 Architecture Diagram

```
┌─────────────┐
│  Frontend   │ (React/Next.js)
│  Dashboard  │
└──────┬──────┘
       │
       │ 1. Initial load: GET /api/dashboard/snapshot
       │ 2. Real-time: WS /ws/dashboard
       │
┌──────▼──────────────────────────────────────┐
│  Dashboard API                              │
│  backend/api/dashboard/                     │
│                                             │
│  routes.py:                                 │
│    - aggregate_portfolio_data()             │
│    - aggregate_positions()                  │
│    - aggregate_signals()                    │
│    - aggregate_risk()                       │
│    - aggregate_system_health()              │
│                                             │
│  websocket.py:                              │
│    - ConnectionManager                      │
│    - subscribe_to_events()                  │
│    - broadcast(event)                       │
└──────┬──────────────────────────────────────┘
       │
       │ Internal HTTP calls (async)
       │
┌──────▼────────────────────────────────────────┐
│  Microservices                                │
├───────────────────────────────────────────────┤
│  Portfolio Intelligence (:8004)               │
│    /api/portfolio/snapshot, /pnl, /drawdown   │
│                                               │
│  AI Engine (:8001)                            │
│    /api/ai/signals/recent (TODO)              │
│                                               │
│  Execution (:8002)                            │
│    /api/execution/positions                   │
│                                               │
│  Risk & Safety (:8003)                        │
│    /api/risk/ess/status                       │
│                                               │
│  Monitoring (:8080)                           │
│    /api/health/services, /alerts              │
└───────────────────────────────────────────────┘
```

---

## ✅ Completion Checklist

- [x] Create data models (`models.py`)
  - [x] DashboardSnapshot (REST)
  - [x] DashboardEvent (WebSocket)
  - [x] Supporting models (Position, Signal, Portfolio, Risk, SystemHealth)
  - [x] Helper functions for event creation

- [x] Create REST endpoint (`routes.py`)
  - [x] GET /api/dashboard/snapshot
  - [x] Aggregation functions for all data sources
  - [x] Error handling and graceful degradation
  - [x] Async parallel fetching

- [x] Create WebSocket handler (`websocket.py`)
  - [x] WS /ws/dashboard
  - [x] ConnectionManager for multiple clients
  - [x] Broadcast mechanism
  - [x] Heartbeat and ping/pong
  - [x] Event subscription placeholder (TODO: EventBus integration)

---

## 🚀 Next Steps (Part 3: Implementation)

1. **Add missing AI Engine endpoint**:
   - Create `GET /api/ai/signals/recent?limit=20` in `microservices/ai_engine/api.py`
   - Store last 20-50 signals in memory or Redis

2. **Integrate with EventBus**:
   - Replace mock event generation in `websocket.py`
   - Subscribe to: `position.updated`, `pnl.updated`, `signal.generated`, `ess.state_changed`, `health.alert`

3. **Test snapshot endpoint**:
   - Start all microservices
   - Call `GET /api/dashboard/snapshot`
   - Verify response structure and data

4. **Test WebSocket**:
   - Connect client (websocat or browser)
   - Trigger internal events (place order, generate signal)
   - Verify broadcast to all clients

5. **Configure service URLs**:
   - Move hardcoded URLs to environment variables
   - Add to `backend/.env` or `docker-compose.yml`

---

## 📝 Notes

- **Service URLs**: Currently hardcoded for Docker network (service names). Need to support localhost for local dev.
- **Signals History**: AI Engine doesn't expose `/signals/recent` yet. Will return empty list until implemented.
- **EventBus**: WebSocket uses mock events for testing. Need to integrate with real EventBus in Part 3.
- **Caching**: Consider adding 5-10 second cache for snapshot endpoint to reduce microservice load.
- **Auth**: No authentication yet. Add JWT verification before production.

---

**Part 2 Status**: ✅ **COMPLETE**  
**Files Created**: 3 (models.py, routes.py, websocket.py)  
**Lines of Code**: ~1000 lines  
**Ready for**: Part 3 (Backend Aggregator Implementation) + Part 4 (Frontend)
