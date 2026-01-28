# P2.8 Portfolio Risk Governor - Authoritative Input Sources

**Date**: 2026-01-28  
**Status**: Production-Grade  
**Component**: P2.8 Portfolio Risk Governor

## Overview

This document defines the authoritative sources of truth for P2.8 Portfolio Risk Governor inputs. These sources must be reliable, continuously updated, and represent the actual system state.

## Input Sources (Ranked by Authority)

### 1. Portfolio State (PRIMARY)

**Authoritative Source**: `quantum:state:portfolio` (Redis hash)

**Fields**:
- `equity_usd` (float): Total portfolio equity in USD
- `drawdown` (float): Current drawdown percentage
- `timestamp` (int): Unix timestamp of last update

**Producer**: Portfolio State Tracker / Data Layer  
**Update Frequency**: Real-time (on position changes)  
**Staleness Tolerance**: 30s (STALE_SEC)  
**Fallback Strategy**: Last-known-good cache (15min max age)

**Production Note**: Currently written by test scripts for testing. In production, this should be continuously updated by:
- Portfolio Ledger Sync service
- Position State Tracker
- Binance account poller

### 2. Position Snapshots (SECONDARY)

**Authoritative Source**: `quantum:position:snapshot:{symbol}` (Redis hashes)

**Purpose**: Identify active symbols requiring budget computation  
**Producer**: Position Ledger / Trade Execution Layer  
**Query Pattern**: `KEYS quantum:position:snapshot:*`  
**Update Frequency**: On position changes

**Fields Used**:
- Symbol extraction only (from key pattern)
- P2.8 uses presence of key to determine active symbols

### 3. Portfolio Heat (REAL-TIME METRIC)

**Authoritative Source**: Heat Gate Prometheus metrics  
**Endpoint**: `http://localhost:8056/metrics`  
**Metric**: `portfolio_heat_composite_normalized`

**Producer**: P2.6 Portfolio Heat Gate  
**Update Frequency**: 10s  
**Fallback**: 0.0 (assume calm if unavailable)

**Note**: Scraped via HTTP, not Redis. This is authoritative for stress computation.

### 4. Cluster Stress (OPTIONAL)

**Authoritative Source**: `quantum:cluster:stress:{cluster_id}` (Redis hash)

**Fields**:
- `stress` (float): Cluster stress factor [0, 1]

**Producer**: Cluster Risk Engine (if deployed)  
**Fallback**: 0.0 (no cluster stress)  
**Status**: Optional - system functions without it

### 5. Volatility Regime (MARKET STATE)

**Authoritative Source**: `quantum:state:market:{symbol}` (Redis hash)

**Fields**:
- `vol_regime` (string): LOW_VOL | NORMAL_VOL | HIGH_VOL | EXTREME_VOL
- `sigma` (float): Backup if vol_regime missing

**Producer**: Market State Engine  
**Fallback**: 0.33 (NORMAL_VOL) if unavailable

## Last-Known-Good Cache Strategy

**Problem**: Portfolio state may become temporarily stale due to:
- Data layer downtime
- Redis connectivity issues
- Upstream service delays

**Solution**: P2.8 implements LKG (Last-Known-Good) cache:

```
1. Fresh state available (<30s old)?
   → Update LKG cache, use fresh state

2. Fresh state stale/missing?
   → Check LKG cache age
   
3. LKG cache valid (<15min)?
   → Use cached state (metric: p28_lkg_cache_used_total)
   
4. LKG cache too old (>15min)?
   → Fail-open, don't write budgets (metric: p28_portfolio_too_old_total)
```

**Benefits**:
- Survives temporary data layer gaps
- Prevents budget expiry during short outages
- Maintains system stability
- Clear metrics for monitoring cache usage

## Budget Hash Output

**Written to**: `quantum:portfolio:budget:{symbol}` (Redis hash)

**TTL**: 300s (5 minutes)  
**Reason**: Long enough to survive temporary gaps, short enough to stay fresh

**Fields**:
```
symbol: str
budget_usd: float (computed budget)
stress_factor: float (composite stress)
equity_usd: float (from portfolio state)
portfolio_heat: float (from Heat Gate)
cluster_stress: float (from cluster engine or 0.0)
vol_regime: float (normalized [0, 1])
mode: str (shadow|enforce)
timestamp: int (budget computation time)
base_risk_pct: float (BASE_RISK_PCT config)
```

**Consumers**:
- P3.2 Governor (Gate 0 / Gate -1)
- Monitoring / Grafana dashboards
- Budget violation stream

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Authoritative Sources                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Portfolio State         Position Snapshots                 │
│  ┌──────────────┐       ┌────────────────┐                 │
│  │ quantum:     │       │ quantum:       │                 │
│  │ state:       │       │ position:      │                 │
│  │ portfolio    │       │ snapshot:*     │                 │
│  └──────┬───────┘       └────────┬───────┘                 │
│         │                        │                          │
│         │     Heat Gate          │                          │
│         │     ┌─────────┐        │                          │
│         │     │ :8056   │        │                          │
│         │     │ /metrics│        │                          │
│         │     └────┬────┘        │                          │
│         │          │             │                          │
│         └──────────┼─────────────┘                          │
│                    │                                         │
│                    ▼                                         │
│        ┌────────────────────────┐                           │
│        │  P2.8 Portfolio Risk   │                           │
│        │  Governor              │                           │
│        │  (with LKG cache)      │                           │
│        └────────────┬───────────┘                           │
│                     │                                        │
│                     ▼                                        │
│         ┌────────────────────────┐                          │
│         │ quantum:portfolio:     │                          │
│         │ budget:{symbol}        │                          │
│         │ (TTL 300s)             │                          │
│         └────────────┬───────────┘                          │
│                      │                                       │
│                      ▼                                       │
│            ┌──────────────────┐                             │
│            │ P3.2 Governor    │                             │
│            │ (Gate 0/-1)      │                             │
│            └──────────────────┘                             │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## Production Readiness Checklist

- [x] LKG cache implemented (15min tolerance)
- [x] Budget hash TTL increased to 300s
- [x] Metrics for cache usage (p28_lkg_cache_used_total, p28_portfolio_too_old_total)
- [x] Fail-open design (missing inputs → no budgets, allow execution)
- [x] Test injection endpoint (P28_TEST_MODE=1 only)
- [ ] **TODO**: Deploy continuous Portfolio State updater
- [ ] **TODO**: Integrate with Position Ledger Sync
- [ ] **TODO**: Alert on p28_portfolio_too_old_total spikes

## Testing

**Test Mode** (P28_TEST_MODE=1):
- Enables `/test/inject_portfolio_state` endpoint
- Allows manual portfolio state injection
- Used for E2E blocking proof without real positions

**Proof Script**: `scripts/proof_p28_enforce_block.sh`

## Monitoring

**Key Metrics**:
- `p28_lkg_cache_used_total`: How often cached state is used
- `p28_portfolio_too_old_total`: Fail-open events due to old data
- `p28_budget_computed_total`: Budgets successfully computed
- `p28_stale_input_total{input_type="portfolio_state"}`: Fresh state unavailable

**Alert Thresholds**:
- `p28_portfolio_too_old_total` > 10/min: Portfolio state producer down
- `p28_lkg_cache_used_total` > 50/min: Fresh state source degraded
- `p28_budget_computed_total` = 0 for 60s: P2.8 loop not running

## Operational Notes

1. **Portfolio State Gap**: If portfolio state is missing, P2.8 will fail-open and NOT write budgets. This is by design - better to allow execution than block with stale data.

2. **Cache Warmup**: On P2.8 restart, LKG cache is empty. First budget computation requires fresh portfolio state. After that, cache provides 15min tolerance.

3. **Manual Override**: For testing, use `POST /test/inject_portfolio_state` (requires P28_TEST_MODE=1).

4. **Path Hygiene**: Ensure P2.8 code is synced between `/home/qt/quantum_trader` and `/opt/quantum` using `scripts/sync_p28_code.sh`.

---
**Document Version**: 1.0  
**Last Updated**: 2026-01-28  
**Author**: AI Agent (Sonnet)  
**Review Required**: Production deployment checklist items
