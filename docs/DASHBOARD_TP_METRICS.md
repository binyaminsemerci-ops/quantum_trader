# TP Metrics Dashboard API Documentation

## Overview

The TP Metrics Dashboard API provides comprehensive analytics for Take-Profit (TP) performance across all trading strategies and symbols. It aggregates data from:

- **TPPerformanceTracker** - Real-time TP metrics tracking
- **tp_profiles_v3** - TP profile configurations  
- **TPOptimizerV3** - Profile optimization recommendations

## Architecture

### Service Layer

**`TPDashboardService`** (`backend/services/dashboard/tp_dashboard_service.py`)

Core service class that aggregates TP data from multiple sources:

```python
from backend.services.dashboard.tp_dashboard_service import get_tp_dashboard_service

service = get_tp_dashboard_service()

# List all tracked pairs
entities = service.list_tp_entities()

# Get complete entry for specific pair
entry = service.get_tp_dashboard_entry("RL_V3", "BTCUSDT")

# Get best/worst performers
summary = service.get_top_best_and_worst(limit=10)
```

### API Layer

**Router:** `backend/api/routes/dashboard_tp.py`  
**Base Path:** `/api/dashboard/tp/`

Three REST endpoints for dashboard consumption.

---

## API Endpoints

### 1. GET `/api/dashboard/tp/entities`

**Purpose:** List all tracked strategy/symbol pairs

**Query Parameters:** None

**Response:** `List[TPDashboardKey]`

```json
[
  {
    "strategy_id": "RL_V3",
    "symbol": "BTCUSDT"
  },
  {
    "strategy_id": "RL_V3",
    "symbol": "ETHUSDT"
  },
  {
    "strategy_id": "RL_V3",
    "symbol": "SOLUSDT"
  }
]
```

**Use Case:** Populate dropdown filters or discovery UI

**Example Request:**
```javascript
const response = await fetch('/api/dashboard/tp/entities');
const entities = await response.json();

// Render dropdown
entities.forEach(entity => {
  addOption(`${entity.strategy_id} - ${entity.symbol}`);
});
```

---

### 2. GET `/api/dashboard/tp/entry`

**Purpose:** Get complete TP analytics for specific strategy/symbol pair

**Query Parameters:**
- `strategy_id` (required): Strategy identifier (e.g., "RL_V3")
- `symbol` (required): Trading symbol (e.g., "BTCUSDT")

**Response:** `TPDashboardEntry`

```json
{
  "key": {
    "strategy_id": "RL_V3",
    "symbol": "BTCUSDT"
  },
  "profile": {
    "profile_id": "TREND_DEFAULT",
    "legs": [
      {
        "label": "TP1",
        "r_multiple": 0.5,
        "size_fraction": 0.15,
        "kind": "SOFT"
      },
      {
        "label": "TP2",
        "r_multiple": 1.0,
        "size_fraction": 0.20,
        "kind": "HARD"
      },
      {
        "label": "TP3",
        "r_multiple": 2.0,
        "size_fraction": 0.30,
        "kind": "HARD"
      }
    ],
    "trailing_profile_id": "TREND_TRAILING",
    "description": "Trend-following: Let profits run with wide trailing"
  },
  "metrics": {
    "tp_hit_rate": 0.64,
    "tp_attempts": 50,
    "tp_hits": 32,
    "tp_misses": 18,
    "avg_r_multiple": 1.56,
    "avg_slippage_pct": 0.008,
    "max_slippage_pct": 0.025,
    "avg_time_to_tp_minutes": 35.2,
    "total_tp_profit_usd": 1580.00,
    "avg_tp_profit_usd": 49.38,
    "premature_exits": 4,
    "missed_opportunities_usd": 180.00,
    "last_updated": "2025-12-10T16:00:00Z"
  },
  "recommendation": {
    "has_recommendation": true,
    "profile_id": "TREND_DEFAULT",
    "suggested_scale_factor": 0.95,
    "reason": "Hit rate 64% slightly above target, bringing TPs closer",
    "confidence": 0.70,
    "direction": "CLOSER"
  }
}
```

**Error Responses:**
- `404` - No metrics found for pair
- `500` - Internal service error

**Use Case:** Detail view for specific trading pair

**Example Request:**
```javascript
const response = await fetch(
  '/api/dashboard/tp/entry?strategy_id=RL_V3&symbol=BTCUSDT'
);

if (response.ok) {
  const entry = await response.json();
  renderDetailView(entry);
} else if (response.status === 404) {
  showMessage("No data available for this pair");
}
```

---

### 3. GET `/api/dashboard/tp/summary`

**Purpose:** Get ranked best and worst performing TP configurations

**Query Parameters:**
- `limit` (optional, default=10): Number of entries for best/worst lists (1-50)

**Response:** `TPDashboardSummary`

```json
{
  "best": [
    {
      "key": {"strategy_id": "RL_V3", "symbol": "BTCUSDT"},
      "profile": { /* ... */ },
      "metrics": {
        "tp_hit_rate": 0.70,
        "tp_attempts": 50,
        "total_tp_profit_usd": 2000.00
      },
      "recommendation": { /* ... */ }
    }
  ],
  "worst": [
    {
      "key": {"strategy_id": "RL_V3", "symbol": "SOLUSDT"},
      "profile": { /* ... */ },
      "metrics": {
        "tp_hit_rate": 0.32,
        "tp_attempts": 25,
        "total_tp_profit_usd": 120.00
      },
      "recommendation": { /* ... */ }
    }
  ],
  "total_entries": 15
}
```

**Ranking Algorithm:**

Performance score = `(hit_rate × 40) + (normalized_r × 30) + (log_profit × 30)`

Where:
- Hit rate: 0-40 points (0.0-1.0 range)
- Avg R multiple: 0-30 points (normalized, 3R = max)
- Total profit: 0-30 points (logarithmic scale)

**Use Case:** Performance overview dashboard

**Example Request:**
```javascript
const response = await fetch('/api/dashboard/tp/summary?limit=5');
const summary = await response.json();

renderTopPerformers(summary.best);
renderBottomPerformers(summary.worst);
```

---

## Data Models

### TPDashboardKey
```typescript
{
  strategy_id: string;  // e.g., "RL_V3"
  symbol: string;       // e.g., "BTCUSDT"
}
```

### TPDashboardProfileLeg
```typescript
{
  label: string;         // "TP1", "TP2", etc.
  r_multiple: number;    // Risk multiple (e.g., 1.0 = 1R)
  size_fraction: number; // Position fraction (0.0-1.0)
  kind: "HARD" | "SOFT"; // Execution type
}
```

### TPDashboardProfile
```typescript
{
  profile_id: string;              // Profile identifier
  legs: TPDashboardProfileLeg[];   // TP legs
  trailing_profile_id: string | null; // Trailing config ID
  description: string;             // Profile description
}
```

### TPDashboardMetrics
```typescript
{
  tp_hit_rate: number;              // Hit rate (0.0-1.0)
  tp_attempts: number;              // Total attempts
  tp_hits: number;                  // Successful hits
  tp_misses: number;                // Misses
  avg_r_multiple: number | null;    // Average R multiple
  avg_slippage_pct: number | null;  // Average slippage %
  max_slippage_pct: number | null;  // Max slippage %
  avg_time_to_tp_minutes: number | null; // Avg time to TP
  total_tp_profit_usd: number | null;    // Total profit (USD)
  avg_tp_profit_usd: number | null;      // Avg profit per hit
  premature_exits: number | null;        // Premature exit count
  missed_opportunities_usd: number | null; // Missed profit
  last_updated: string | null;      // ISO timestamp
}
```

### TPDashboardRecommendation
```typescript
{
  has_recommendation: boolean;
  profile_id: string | null;
  suggested_scale_factor: number | null;  // 0.9 = closer, 1.1 = further
  reason: string | null;
  confidence: number | null;  // 0.0-1.0
  direction: "CLOSER" | "FURTHER" | "NO_CHANGE" | null;
}
```

---

## Frontend Integration Guide

### Main Dashboard Table

**Columns:**
- Strategy
- Symbol
- Hit Rate
- Attempts
- Avg R
- Slippage
- Profit
- Recommendation Badge

**Implementation:**
```javascript
async function loadDashboardTable() {
  const response = await fetch('/api/dashboard/tp/entities');
  const entities = await response.json();
  
  for (const entity of entities) {
    const entryResponse = await fetch(
      `/api/dashboard/tp/entry?` +
      `strategy_id=${entity.strategy_id}&symbol=${entity.symbol}`
    );
    const entry = await entryResponse.json();
    
    renderTableRow({
      strategy: entry.key.strategy_id,
      symbol: entry.key.symbol,
      hitRate: (entry.metrics.tp_hit_rate * 100).toFixed(1) + '%',
      attempts: entry.metrics.tp_attempts,
      avgR: entry.metrics.avg_r_multiple?.toFixed(2) + 'R',
      slippage: entry.metrics.avg_slippage_pct 
        ? (entry.metrics.avg_slippage_pct * 100).toFixed(2) + '%' 
        : 'N/A',
      profit: entry.metrics.total_tp_profit_usd 
        ? '$' + entry.metrics.total_tp_profit_usd.toFixed(2)
        : 'N/A',
      hasRecommendation: entry.recommendation.has_recommendation
    });
  }
}
```

### Detail Panel (Expandable Row)

**Show:**
- TP Profile legs table
- Trailing configuration
- Performance metrics breakdown
- Optimization recommendation (if available)

**Implementation:**
```javascript
function renderDetailPanel(entry) {
  // Render TP legs
  entry.profile.legs.forEach(leg => {
    addLegRow({
      label: leg.label,
      target: `${leg.r_multiple}R`,
      size: `${(leg.size_fraction * 100).toFixed(0)}%`,
      type: leg.kind
    });
  });
  
  // Render trailing
  if (entry.profile.trailing_profile_id) {
    showTrailing Info(entry.profile.trailing_profile_id);
  }
  
  // Render recommendation
  if (entry.recommendation.has_recommendation) {
    showRecommendationBadge({
      direction: entry.recommendation.direction,
      scale: entry.recommendation.suggested_scale_factor,
      reason: entry.recommendation.reason,
      confidence: (entry.recommendation.confidence * 100).toFixed(0) + '%'
    });
  }
}
```

### Performance Overview Cards

**Top Performers & Bottom Performers**

**Implementation:**
```javascript
async function loadPerformanceOverview() {
  const response = await fetch('/api/dashboard/tp/summary?limit=5');
  const summary = await response.json();
  
  // Render top performers
  renderCard('top-performers', {
    title: `Top ${summary.best.length} Performers`,
    entries: summary.best.map(entry => ({
      label: `${entry.key.strategy_id}/${entry.key.symbol}`,
      hitRate: (entry.metrics.tp_hit_rate * 100).toFixed(1) + '%',
      profit: '$' + entry.metrics.total_tp_profit_usd.toFixed(2)
    }))
  });
  
  // Render bottom performers
  renderCard('bottom-performers', {
    title: `Bottom ${summary.worst.length} Performers`,
    entries: summary.worst.map(entry => ({
      label: `${entry.key.strategy_id}/${entry.key.symbol}`,
      hitRate: (entry.metrics.tp_hit_rate * 100).toFixed(1) + '%',
      needsAttention: entry.recommendation.has_recommendation
    }))
  });
}
```

---

## Performance Considerations

### Caching

The service layer uses singleton instances to minimize overhead:

```python
# Reuses TPPerformanceTracker singleton
service = get_tp_dashboard_service()
```

### Optimization Tips

1. **Batch Loading:** Use `/entities` + `/entry` for initial load
2. **Polling:** Poll `/summary` endpoint periodically (e.g., every 5 minutes)
3. **Detail-on-Demand:** Load `/entry` only when user expands detail view
4. **Local Caching:** Cache entity list in frontend to reduce API calls

### Rate Limiting

Recommended API call patterns:
- `/entities`: Once on page load
- `/entry`: On-demand per user action
- `/summary`: Every 5-10 minutes background refresh

---

## Testing

### Service Tests

Location: `tests/services/test_tp_dashboard_service.py`

Coverage:
- ✅ list_tp_entities() - multiple pairs, empty
- ✅ get_tp_dashboard_entry() - complete entry, not found, errors
- ✅ get_top_best_and_worst() - ranking, limits, empty

Run: `pytest tests/services/test_tp_dashboard_service.py -v`

### API Tests

Location: `tests/api/test_dashboard_tp_routes.py`

Coverage:
- ✅ GET /entities - success, empty, errors
- ✅ GET /entry - success, not found, validation, errors
- ✅ GET /summary - success, limits, validation, errors

Run: `pytest tests/api/test_dashboard_tp_routes.py -v`

---

## Example Dashboard Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  TP Performance Dashboard                                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐                   │
│  │ Top Performers   │  │ Bottom Performers│                   │
│  │ - RL_V3/BTCUSDT │  │ - RL_V3/SOLUSDT  │                   │
│  │   70% | $2000   │  │   32% | $120     │                   │
│  │ - RL_V3/ETHUSDT │  │ - RL_V3/DOTUSDT  │                   │
│  │   58% | $920    │  │   40% | $200     │                   │
│  └──────────────────┘  └──────────────────┘                   │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Strategy/Symbol Table                                          │
│  ┌────────┬──────────┬──────┬──────┬─────┬─────────┬────────┐│
│  │Strategy│Symbol    │Hit % │Att   │Avg R│Slippage │Profit  ││
│  ├────────┼──────────┼──────┼──────┼─────┼─────────┼────────┤│
│  │RL_V3   │BTCUSDT [▼]│ 64%  │  50  │1.56R│  0.8%   │$1,580  ││
│  │        │  TP1: 0.5R @ 15% (SOFT)                          ││
│  │        │  TP2: 1.0R @ 20% (HARD)                          ││
│  │        │  TP3: 2.0R @ 30% (HARD)                          ││
│  │        │  🔽 Recommendation: Bring TPs 5% closer (70%)    ││
│  ├────────┼──────────┼──────┼──────┼─────┼─────────┼────────┤│
│  │RL_V3   │ETHUSDT   │ 58%  │  32  │1.78R│  1.2%   │  $920  ││
│  │RL_V3   │SOLUSDT   │ 32%  │  25  │3.13R│  1.5%   │  $120  ││
│  └────────┴──────────┴──────┴──────┴─────┴─────────┴────────┘│
└─────────────────────────────────────────────────────────────────┘
```

---

## Next Steps

### Future Enhancements

1. **Filtering:** Add query params for strategy/symbol filtering on `/summary`
2. **Regime-Aware:** Support regime parameter for profile lookup
3. **Historical Trends:** Add time-series data endpoints
4. **Sorting:** Custom sort parameters (by hit_rate, profit, etc.)
5. **Export:** CSV/JSON export functionality
6. **WebSocket:** Real-time updates as new TP hits occur

### Integration Checklist

- [ ] Implement frontend dashboard using these endpoints
- [ ] Add error handling and loading states
- [ ] Implement local caching for entity list
- [ ] Add periodic refresh for summary data
- [ ] Create detail view component for expanded rows
- [ ] Add filters and search functionality
- [ ] Implement sorting and pagination
- [ ] Add export/download features

---

## Support

For questions or issues:
- Service layer: `backend/services/dashboard/tp_dashboard_service.py`
- API routes: `backend/api/routes/dashboard_tp.py`
- Tests: `tests/services/` and `tests/api/`
