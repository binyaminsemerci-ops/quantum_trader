# P2.6 Components — Disambiguation Guide

**Last Updated**: 2026-01-27 06:20 UTC

---

## ⚠️ IMPORTANT: Two Different P2.6 Components

The "P2.6" label refers to **TWO SEPARATE COMPONENTS** with different purposes and deployment status:

---

## 1️⃣ P2.6 Portfolio Gate

**File**: `microservices/portfolio_gate/main.py` (539 lines)  
**Service**: `quantum-portfolio-gate.service`  
**Port**: 8047 (Prometheus metrics)

### Status
✅ **LIVE AND OPERATIONAL** (deployed with P2.7 on 2026-01-27)

### Purpose
Intelligent portfolio-level gating that integrates **P2.7 cluster stress** to make permit decisions.

### Architecture
```
P2.5 Harvest Kernel
   ↓ quantum:stream:harvest.proposal
P2.6 Portfolio Gate  ← reads P2.7 cluster stress
   ↓ quantum:stream:portfolio.gate
   ↓ quantum:permit:p26:{plan_id}
P3.1 Apply Layer
```

### Key Functions
- Reads cluster stress from P2.7: `quantum:portfolio:cluster_state`
- Computes portfolio metrics (net notional, gross notional, position count)
- Issues permits for Apply Layer based on:
  - Cluster stress (fail-open: proxy correlation if P2.7 unavailable)
  - Portfolio metrics (cooling for overcrowded states)
  - Cooldown gates (symbol-level)

### Metrics (Port 8047)
```bash
# Cluster integration
p26_cluster_stress_used=1    # Using cluster-based K (not proxy)
p26_cluster_stress=0.748     # Current cluster stress

# Portfolio state
p26_portfolio_net_notional_usdt
p26_portfolio_gross_notional_usdt
p26_portfolio_num_positions

# Decisions
p26_decision_total{action="PERMIT"}
p26_decision_total{action="HOLD"}
```

### VPS Evidence
```bash
systemctl status quantum-portfolio-gate
# Active: active (running) since Mon 2026-01-27 05:22:50 UTC

journalctl -u quantum-portfolio-gate -n 5
# [INFO] P2.6: K=0.748 (cluster)
# [INFO] p26_cluster_stress_used=1
```

### Documentation
- [P2_7_PRODUCTION_MONITORING.md](P2_7_PRODUCTION_MONITORING.md) — P2.6 integration with P2.7
- [P2_7_LIVE_VERIFICATION.md](P2_7_LIVE_VERIFICATION.md) — Cluster stress usage verified

---

## 2️⃣ P2.6 Portfolio Heat Gate

**File**: `microservices/portfolio_heat_gate/main.py` (420 lines)  
**Service**: `quantum-portfolio-heat-gate.service`  
**Port**: 8056 (Prometheus metrics)

### Status
⏸️ **SHADOW MODE** (deployed 2026-01-26, logging only, NOT affecting trading)

### Purpose
Heat-based exit calibration that prevents premature FULL_CLOSE when portfolio is "cold".

### Architecture (When Activated)
```
P2.5 Harvest Kernel
   ↓ harvest.proposal
P2.6 Portfolio Heat Gate  ← calibrates based on heat
   ↓ harvest.calibrated
P3.1 Apply Layer (reads calibrated)
```

### Key Functions
- Calculates Portfolio Heat: `Σ(|notional| * sigma) / equity`
- Downgrades aggressive exits based on heat buckets:
  - **COLD** (heat < 0.25): FULL_CLOSE → PARTIAL_25
  - **WARM** (0.25 ≤ heat < 0.65): FULL_CLOSE → PARTIAL_75
  - **HOT** (heat ≥ 0.65): FULL_CLOSE allowed
- Currently logs comparisons but **does NOT modify proposals**

### Metrics (Port 8056)
```bash
# Heat calculation
p26_heat_value=0.8825           # Current portfolio heat
p26_bucket{state="HOT"}=1.0     # In HOT bucket

# Downgrades (shadow mode: counts hypothetical downgrades)
p26_actions_downgraded_total=0  # No proposals processed yet
```

### VPS Evidence
```bash
systemctl status quantum-portfolio-heat-gate
# Active: active (running) since Mon 2026-01-26 23:59:50 UTC

cat /etc/quantum/portfolio-heat-gate.env | grep MODE
# P26_MODE=shadow    ← NOT enforce (not affecting trading)

curl http://localhost:8056/metrics | grep p26_heat
# p26_heat_value 0.8825
# p26_bucket{state="HOT"} 1.0
```

### Current Blocker
🔴 **Apply Layer integration incomplete** — Heat Gate writes to stream, but Apply Layer reads hash keys. Requires patching before activation.

### Documentation
- [P2_6_PORTFOLIO_HEAT_GATE.md](P2_6_PORTFOLIO_HEAT_GATE.md) — Heat Gate specification
- [P2_6_HEAT_GATE_ACTIVATION_PLAN.md](P2_6_HEAT_GATE_ACTIVATION_PLAN.md) — Activation roadmap
- [TESTNET_FUND_CAPS_AND_P26_SHADOW_DEPLOYED.md](TESTNET_FUND_CAPS_AND_P26_SHADOW_DEPLOYED.md) — Shadow deployment report

---

## 🔍 How to Tell Them Apart

| Aspect | P2.6 Portfolio Gate | P2.6 Portfolio Heat Gate |
|--------|---------------------|--------------------------|
| **File** | `portfolio_gate/main.py` | `portfolio_heat_gate/main.py` |
| **Service** | `quantum-portfolio-gate` | `quantum-portfolio-heat-gate` |
| **Port** | 8047 | 8056 |
| **Status** | ✅ LIVE | ⏸️ SHADOW MODE |
| **Input** | P2.7 cluster stress | Portfolio heat (notional * sigma) |
| **Output** | Permits for Apply Layer | Calibrated proposals (shadow only) |
| **Integration** | P2.7 cluster stress | P2.5 Harvest (when activated) |
| **Deployed** | 2026-01-27 (with P2.7) | 2026-01-26 (shadow) |
| **Ops Log** | OPS-2026-01-27-010 | TESTNET_FUND_CAPS (no ops entry) |

---

## 📋 Quick Reference Commands

### Check Portfolio Gate (LIVE)
```bash
# VPS commands
systemctl status quantum-portfolio-gate
journalctl -u quantum-portfolio-gate -n 20
curl http://localhost:8047/metrics | grep p26_

# Look for
p26_cluster_stress_used=1    # Using P2.7 clusters
p26_cluster_stress=0.748     # Cluster stress value
```

### Check Heat Gate (SHADOW)
```bash
# VPS commands
systemctl status quantum-portfolio-heat-gate
journalctl -u quantum-portfolio-heat-gate -n 20
curl http://localhost:8056/metrics | grep p26_

# Config
cat /etc/quantum/portfolio-heat-gate.env | grep MODE
# Should show: P26_MODE=shadow (not enforce)
```

---

## 🎯 Summary for Operations

**When someone says "P2.6"**, ask:
- **Portfolio Gate** (cluster stress integration)? → LIVE, operational, deployed with P2.7
- **Heat Gate** (exit calibration)? → SHADOW MODE, not affecting trading, needs activation

**Deployment Status**:
- ✅ **Portfolio Gate**: Deployed and verified (OPS-2026-01-27-010)
- ⏸️ **Heat Gate**: Deployed but shadow mode (awaiting activation plan execution)

**Next Steps**:
- Portfolio Gate: Monitor cluster stress integration
- Heat Gate: Collect shadow data, patch Apply Layer, activate when ready

---

**Document Created**: 2026-01-27 06:20 UTC  
**Purpose**: Resolve P2.6 naming confusion
