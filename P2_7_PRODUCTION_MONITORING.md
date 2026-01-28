# P2.7 Portfolio Clusters - Production Monitoring Guide

**Status:** ✅ FULLY OPERATIONAL (2026-01-27)

## End-to-End Verification Completed

### ✅ P2.7 Correlation/Clusters READY
- `p27_corr_ready=1` - Correlation matrix computing successfully
- Clusters detected and updated continuously (60s cadence)
- ClusterStress being calculated and written to Redis

### ✅ Cluster State in Redis
- Key: `quantum:portfolio:cluster_state`
- Fields: `cluster_stress`, `clusters_count`, `updated_ts`
- Freshness: Updated every 60s when P2.7 active

### ✅ P2.6 Switched from Proxy to Cluster
- Logs show: `"K=0.748 (cluster), stress=0.962"`
- Metric: `p26_cluster_stress_used=1` (1=cluster, 0=proxy)
- Integration verified with live proposals

### ✅ Fallback Path Intact
- If P2.7 becomes stale (>120s) → P2.6 reverts to proxy
- Counter: `p26_cluster_fallback_total` tracks fallback events
- Graceful degradation ensures continuous operation

---

## Minimal "Live Health" Metrics Set

Monitor these 5 metrics for P2.7 production health:

### 1. P2.7 Update Rate
```promql
rate(p27_updates_total[10m]) > 0
```
**Meaning:** P2.7 is actively computing and updating cluster state  
**Alert if:** `== 0` for >5 minutes (P2.7 stuck or crashed)

### 2. Correlation Matrix Ready
```promql
p27_corr_ready == 1
```
**Meaning:** P2.7 has sufficient data (11+ points) to compute correlations  
**Alert if:** `== 0` for >15 minutes (data collection issue)

### 3. Snapshot Freshness
```promql
max(p26_snapshot_age_seconds) < 300
```
**Meaning:** Position snapshots are recent (<5 minutes old)  
**Alert if:** `> 300` (upstream data pipeline issue)

### 4. P2.6 Cluster Usage (When Active)
```promql
increase(p26_cluster_stress_used[10m]) > 0
```
**Meaning:** P2.6 is using cluster stress from P2.7 when processing proposals  
**Note:** Only increments when proposals are being processed  
**Alert if:** `== 0` AND proposals are being processed (integration broken)

### 5. Fallback Events (Should be Rare)
```promql
increase(p26_cluster_fallback_total[30m]) == 0
```
**Meaning:** P2.6 is NOT falling back to proxy correlation  
**Alert if:** `> 0` consistently (P2.7 unstable or data staleness issue)

---

## Critical Alerts

### 🚨 P2.7 Stale/Stopped
```promql
rate(p27_updates_total[10m]) == 0
```
**Impact:** P2.6 will fallback to proxy correlation (less accurate)  
**Action:**
1. Check service: `systemctl status quantum-portfolio-clusters`
2. Check logs: `journalctl -u quantum-portfolio-clusters -n 100`
3. Verify Redis connection: `redis-cli PING`
4. Check data pipeline: `redis-cli HGETALL quantum:position:snapshot:BTCUSDT`

### 🚨 P2.6 Falling Back Too Often
```promql
increase(p26_cluster_fallback_total[30m]) > 0
```
**Impact:** Stress computation using less accurate proxy instead of real clusters  
**Root Causes:**
- P2.7 service crashed or stuck
- Redis cluster_state not being written
- cluster_state `updated_ts` too old (>120s)
- Network issues between P2.6 and Redis

**Action:**
1. Check P2.7 metrics: `curl http://127.0.0.1:8048/metrics | grep p27_`
2. Check cluster state: `redis-cli HGETALL quantum:portfolio:cluster_state`
3. Verify freshness: Compare `updated_ts` with current epoch
4. Check P2.6 logs for fallback reason

---

## Warmup Metrics (Operational Transparency)

### Buffer Fill Progress
```promql
p27_points_per_symbol{symbol="BTCUSDT"}
p27_points_per_symbol{symbol="ETHUSDT"}
p27_min_points_per_symbol
```
**Purpose:** Track data collection after service restart  
**Threshold:** Need 11+ points for correlation computation (10 returns)  
**Time:** ~11 minutes after restart (1 point/minute at 60s cadence)

### Warmup Complete When:
- `p27_min_points_per_symbol >= 10` (actually need 11+ for 10 returns)
- `p27_corr_ready == 1`
- `p27_updates_total > 0`

---

## Service Dependencies

### Upstream (P2.7 depends on):
- **Redis** - Must be available on localhost:6379
- **Position Snapshots** - Keys: `quantum:position:snapshot:{SYMBOL}`
  - Fields needed: `position_amt`, `mark_price`, `ts_epoch`
  - Freshness: Should update every 60s or less
- **Allowlist Symbols** - Currently: BTCUSDT, ETHUSDT, SOLUSDT
  - At least 2 symbols must have active positions for clustering

### Downstream (Services that use P2.7):
- **P2.6 Portfolio Gate** (`quantum-portfolio-gate.service`)
  - Reads: `quantum:portfolio:cluster_state` from Redis
  - Fallback: Uses proxy correlation if cluster_state stale (>120s)
  - Integration metric: `p26_cluster_stress_used` (0=proxy, 1=cluster)

---

## Operational Notes

### Service Restart Impact
- **In-memory buffers cleared** - All price history lost
- **Warmup required** - Need 11+ data points before `p27_corr_ready=1`
- **Warmup time** - Approximately 11 minutes (60s update cadence)
- **P2.6 behavior during warmup** - Automatically uses proxy fallback

### Normal Operation
- **Update frequency:** Every 60 seconds
- **Lookback window:** 360 minutes (6 hours)
- **Clustering threshold:** Correlation >= 0.70
- **Port:** 8048 (Prometheus metrics)
- **Redis stream audit:** `quantum:stream:portfolio.cluster_state`

### Expected Metrics (Healthy State)
```
p27_corr_ready = 1
p27_symbols_in_matrix = 2-3 (depending on active positions)
p27_clusters_count = 1-3 (typically 1 for correlated crypto markets)
p27_cluster_stress_sum = 0.0-1.0 (portfolio cluster stress)
p27_updates_total = increasing counter
p27_fail_closed_total = 0 or low (rare edge cases)
```

---

## Quick Diagnostic Commands

### Check P2.7 Health
```bash
# Service status
systemctl status quantum-portfolio-clusters

# Recent logs
journalctl -u quantum-portfolio-clusters -n 50

# Metrics snapshot
curl -s http://127.0.0.1:8048/metrics | grep p27_

# Specific metrics
curl -s http://127.0.0.1:8048/metrics | grep -E "p27_(corr_ready|updates_total|clusters_count|cluster_stress_sum)"
```

### Check P2.6 Integration
```bash
# P2.6 cluster metrics
curl -s http://127.0.0.1:8047/metrics | grep p26_cluster

# Recent P2.6 logs showing cluster usage
journalctl -u quantum-portfolio-gate -n 50 | grep cluster

# Redis cluster state
redis-cli HGETALL quantum:portfolio:cluster_state

# Check freshness
redis-cli HGET quantum:portfolio:cluster_state updated_ts
date +%s  # Compare with current epoch
```

### Verify Data Pipeline
```bash
# Check position snapshots exist
redis-cli KEYS "quantum:position:snapshot:*"

# Inspect snapshot structure
redis-cli HGETALL quantum:position:snapshot:ETHUSDT

# Check snapshot freshness
redis-cli HGET quantum:position:snapshot:ETHUSDT ts_epoch
```

---

## Rollback Procedure (If Needed)

```bash
# Stop P2.7
systemctl stop quantum-portfolio-clusters
systemctl disable quantum-portfolio-clusters

# P2.6 will automatically fallback to proxy
# Verify fallback: journalctl -u quantum-portfolio-gate | grep proxy

# Revert code (if needed)
git revert b556c2d8  # P2.7 deployment commit
git push origin main

# P2.6 continues operating normally with proxy correlation
```

---

## Performance Characteristics

### CPU/Memory (Observed)
- **Memory:** ~18 MB (steady state)
- **CPU:** <1% (60s update cadence)
- **Threads:** 2 (main + metrics server)

### Data Volume
- **Ring buffers:** 360 data points × 3 symbols = 1080 PricePoint objects
- **Memory per point:** ~32 bytes (float + int)
- **Total buffer memory:** ~35 KB (negligible)

### Correlation Computation Complexity
- **Symbols:** N = 2-3
- **Pairwise correlations:** N(N-1)/2 = 1-3 pairs
- **Per correlation:** O(M) where M = buffer length (360 max)
- **Total:** O(N² × M) ≈ 3000 operations per update (trivial)

---

## Version Info

- **Implementation:** commit `b556c2d8` (2026-01-27)
- **Service:** `quantum-portfolio-clusters.service`
- **Config:** `/etc/quantum/portfolio-clusters.env`
- **Binary:** `/usr/bin/python3 -m microservices.portfolio_clusters.main`
- **Working Dir:** `/home/qt/quantum_trader`
- **User:** `qt`

---

## Related Documentation

- **P2.6 Portfolio Gate:** See `P2_6_PRODUCTION_GUIDE.md`
- **Architecture:** See `AI_COMPLETE_MODULE_OVERVIEW.md`
- **Deployment:** See `ops/p27_deploy_and_proof.sh`
- **Spec:** P2.7 LOCKED SPEC (correlation matrix + capital clustering)
