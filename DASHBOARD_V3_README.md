# 📊 Dashboard V3.0 - Real-Time Trading Dashboard

**EPIC:** DASHBOARD-V3-001  
**Status:** ✅ Phase 9 Complete (75% - Real-time WebSocket integration)  
**Created:** December 4, 2025

---

## 🎯 Overview

Quantum Trader Dashboard V3.0 is a modern, tabbed interface for real-time trading operations monitoring. Built with **Next.js**, **React**, **TypeScript**, and **Tailwind CSS**, it provides comprehensive visibility into:

- **GO-LIVE status** and risk state
- **Live positions** and orders
- **RiskGate decisions** and ESS triggers
- **System health** and stress scenarios
- **Real-time WebSocket updates** with polling fallback

---

## 🏗️ Architecture

### Backend-for-Frontend (BFF) Layer

**File:** `backend/api/dashboard/bff_routes.py` (332 lines)

Aggregates data from 5 microservices:
- `portfolio-intelligence:8004` - PnL, equity, exposure
- `ai-engine:8001` - Signals, strategies
- `execution:8002` - Orders, positions
- `risk-safety:8003` - RiskGate, ESS, VaR/ES
- `monitoring-health:8080` - System health, exchanges

**Endpoints:**
```
GET  /api/dashboard/overview     # GO-LIVE, PnL, risk, ESS
GET  /api/dashboard/trading      # Positions, orders, signals
GET  /api/dashboard/risk         # RiskGate, ESS, VaR/ES
GET  /api/dashboard/system       # Services, exchanges, failover
POST /api/dashboard/stress/run_all  # Run stress scenarios
```

### Frontend Components

**Main Page:** `frontend/pages/index.tsx`  
5-tab navigation: Overview | Trading | Risk | System | Classic

**Tab Components:**
1. **OverviewTab.tsx** (289 lines) - GO-LIVE, global PnL, risk state, ESS status
2. **TradingTab.tsx** (239 lines) - Positions, orders, signals, strategies
3. **RiskTab.tsx** (262 lines) - RiskGate decisions, ESS triggers, VaR/ES
4. **SystemTab.tsx** (308 lines) - Services health, exchanges, stress runner

**Real-Time Hook:** `frontend/hooks/useDashboardStream.ts` (183 lines)
- WebSocket connection to `/ws/dashboard`
- Polling fallback every 5 seconds
- Live counters: positions, blocked trades, failovers

### WebSocket Integration (Phase 9)

**Components with Real-Time Updates:**
- **TopBar** - Live position count badge
- **OverviewTab** - GO-LIVE, risk state, ESS, PnL with live indicators
  - Shows blocked trades count (last 5m)
  - Shows failover events (last 5m)
  - Green pulse indicator when WebSocket connected

**Data Flow:**
```
WebSocket (/ws/dashboard)
    ↓
useDashboardStream hook
    ↓
React components (TopBar, OverviewTab)
    ↓
Live UI updates
```

**Fallback Strategy:**
- WebSocket connected → Real-time updates
- WebSocket disconnected → Poll `/api/dashboard/overview` every 5s
- Error handling → Display error message with retry

---

## 🚀 Quick Start

### 1. Start Backend
```powershell
# Option A: Docker
docker-compose up backend -d

# Option B: Direct
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Start Frontend
```powershell
cd frontend
npm install
npm run dev
```

### 3. Access Dashboard
- **Frontend:** http://localhost:3000
- **Backend API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs

---

## 📋 Features by Tab

### 🏠 Overview Tab
- **GO-LIVE Status** - Active/Inactive badge with environment
- **Risk State** - OK/WARNING/CRITICAL with blocked trades counter
- **ESS Status** - Emergency Stop System state with triggers
- **Daily PnL** - Dollar amount and percentage with failover alerts
- **Equity Breakdown** - Total, weekly, monthly PnL
- **Exposure** - Per-exchange net exposure list
- **Real-Time:** WebSocket updates every event, 5s polling fallback

### 📈 Trading Tab
- **Open Positions** - 7-column table (symbol, side, size, entry, current, PnL, exchange)
- **Recent Orders** - Last 50 orders with status badges
- **Recent Signals** - Last 20 AI signals with confidence scores
- **Active Strategies** - Per-account strategy cards
- **Refresh:** 3 seconds (high-frequency trading data)

### 🛡️ Risk Tab
- **RiskGate Decisions** - Total/Allow/Scale/Block stats with percentages
- **ESS Triggers** - Last 24h triggers with timestamps and reasons
- **Drawdown per Profile** - Progress bars by capital profile (micro/low/normal/agg)
- **VaR/ES Snapshot** - Value-at-Risk and Expected Shortfall (95%/99%)
- **Refresh:** 10 seconds (risk metrics update slower)

### ⚙️ System Tab
- **Services Health** - 4-column grid with UP/DOWN/DEGRADED status
- **Exchanges Health** - Binance/Bybit status with latency
- **Failover Events** - Recent exchange failovers with reasons
- **Stress Scenarios** - Recent runs with PASS/FAIL badges
- **Run All Button** - Trigger all stress tests (POST endpoint)
- **Refresh:** 15 seconds (system metrics)

---

## 🔌 WebSocket Events

**Connection:** `ws://localhost:8000/ws/dashboard`

**Event Types:**
```typescript
type DashboardEvent = 
  | { type: 'connected' }
  | { type: 'disconnected' }
  | { type: 'error', message: string }
  | { type: 'snapshot', data: any }
  | { type: 'update', data: any }
```

**Real-Time Metrics:**
- `go_live_active` (boolean)
- `risk_state` (OK|WARNING|CRITICAL)
- `ess_active` (boolean)
- `open_positions_count` (number)
- `blocked_trades_last_5m` (number)
- `failovers_last_5m` (number)
- `daily_pnl` (number)
- `daily_pnl_pct` (number)

---

## 🎨 UI/UX Features

### Color Coding
- **Green** - Success, OK, ACTIVE (GO-LIVE), INACTIVE (ESS)
- **Yellow** - Warning, scaled trades, failover events
- **Red** - Danger, CRITICAL, ACTIVE (ESS), blocked trades
- **Gray** - Inactive, neutral states

### Auto-Refresh Intervals
- **Trading:** 3s (positions, orders change frequently)
- **Overview:** 5s (balanced between real-time and load)
- **Risk:** 10s (VaR/ES updates slower)
- **System:** 15s (health metrics are stable)

### Loading States
- Skeleton cards with `animate-pulse`
- Prevents layout shift during data load
- Shows 4 skeleton cards in grid

### Error Handling
- Centered error messages with icons
- Retry buttons for manual refresh
- Graceful degradation (WebSocket → polling)

---

## 📦 File Structure

```
backend/
  api/dashboard/
    bff_routes.py          # 332 lines - BFF aggregation layer

frontend/
  pages/
    index.tsx              # Enhanced with 5-tab navigation
  
  components/
    TopBar.tsx             # Updated with live position count
    dashboard/
      OverviewTab.tsx      # 289 lines - Global status
      TradingTab.tsx       # 239 lines - Positions & orders
      RiskTab.tsx          # 262 lines - Risk metrics
      SystemTab.tsx        # 308 lines - System health
  
  hooks/
    useDashboardStream.ts  # 183 lines - Real-time hook
  
  lib/
    websocket.ts           # Existing WebSocket client
```

---

## 🧪 Testing

### Backend Endpoints
```powershell
# Test overview endpoint
curl http://localhost:8000/api/dashboard/overview

# Test trading endpoint
curl http://localhost:8000/api/dashboard/trading

# Test stress scenarios
curl -X POST http://localhost:8000/api/dashboard/stress/run_all
```

### Frontend Components
```bash
# Run frontend tests (when available)
cd frontend
npm test
```

### WebSocket Connection
```javascript
// Browser console
const ws = new WebSocket('ws://localhost:8000/ws/dashboard');
ws.onmessage = (e) => console.log('Event:', JSON.parse(e.data));
```

---

## 🔄 Integration with Existing System

### Preserved Features
- **Classic Dashboard** - Original view available as "Classic" tab
- **WebSocket Connection** - Reuses existing `dashboardWebSocket` client
- **Store Pattern** - Maintains `useDashboardStore` for compatibility
- **API Patterns** - Follows existing fetch/error handling conventions

### New Additions
- **BFF Router** - Registered in `backend/main.py` line 2295
- **Tab Navigation** - Enhanced `index.tsx` with state management
- **Real-Time Hook** - New `useDashboardStream` for live updates
- **4 New Endpoints** - Overview, Trading, Risk, System aggregators

---

## 📊 Progress Tracking

**Completed Phases (9/12 - 75%):**
- ✅ Phase 1: Discovery (existing API & frontend)
- ✅ Phase 2: BFF layer (5 endpoints)
- ✅ Phase 3: WebSocket verification
- ✅ Phase 4: Frontend shell (tab navigation)
- ✅ Phase 5: OverviewTab component
- ✅ Phase 6: TradingTab component
- ✅ Phase 7: RiskTab component
- ✅ Phase 8: SystemTab component
- ✅ Phase 9: Real-time WebSocket hook

**Remaining Phases:**
- ⏳ Phase 10: Styling & UX polish (dark mode, responsive, spacing)
- ⏳ Phase 11: Tests & sanity checks (endpoint tests, component tests)
- ⏳ Phase 12: Final documentation & deployment guide

---

## 🛠️ Configuration

### Environment Variables
```bash
# Frontend (.env.local)
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000

# Backend (.env)
BINANCE_TESTNET=true
MULTI_EXCHANGE_ENABLED=true
GO_LIVE_ACTIVE=false  # Marker file controls actual state
```

### Feature Flags
- GO-LIVE system: `backend/go_live/go_live.active` marker file
- ESS threshold: `config/ess.yaml` (default: -5% daily loss)
- Max positions: `RM_MAX_CONCURRENT_TRADES=20` (.env)

---

## 📚 Next Steps

1. **Phase 10 - Styling:**
   - Consistent Tailwind spacing (`space-y-6`, `p-4`)
   - Dark mode verification
   - Responsive breakpoints (mobile-first)
   - Environment badge (STAGING/PRODUCTION)

2. **Phase 11 - Testing:**
   - FastAPI test client for BFF endpoints
   - React Testing Library for components
   - WebSocket integration test
   - Error handling validation

3. **Phase 12 - Documentation:**
   - API reference (Swagger/OpenAPI)
   - Component documentation (Storybook)
   - Deployment guide (Docker + K8s)
   - User manual

---

## 🤝 Contributing

Dashboard V3.0 follows existing codebase patterns:
- **TypeScript** - Strict mode, explicit types
- **React Hooks** - Functional components, custom hooks
- **Tailwind CSS** - Utility-first styling
- **Error Handling** - Try-catch with user-friendly messages
- **Auto-Refresh** - setInterval with cleanup

---

## 📄 License

Part of Quantum Trader system - proprietary hedge fund trading OS.

---

**Last Updated:** December 4, 2025  
**Phase:** 9/12 Complete (75%)  
**Status:** Real-time WebSocket integration operational ✅
