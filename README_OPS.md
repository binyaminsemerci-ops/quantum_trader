# Quantum Trader — VPS Architecture & Infrastructure
**Scope:** VPS runtime only (NOT local dev)  
**Host:** Hetzner VPS `46.224.116.254` (`quantumtrader-prod-1`)  
**OS:** Linux Ubuntu (kernel 6.8.0-90-generic)  
**Service manager:** systemd (migrated from Docker, January 2026)  
**Ingress:** NGINX 80/443  
**Core Bus:** Redis 6379  
**Repo path (VPS):** `/home/qt/quantum_trader` (owner: `qt`)  
**Venv path (VPS):** `/opt/quantum/venvs/ai-engine/`  
**Env files (VPS):** `/etc/quantum/*.env`  

![VPS Topology](docs/diagrams/quantum_trader_vps_topology.png)

See also:
- [`OWNERSHIP_MAP.md`](OWNERSHIP_MAP.md) — per-layer ownership, streams, failure lookup
- [`SONNET_FORENSIC_PROMPT.txt`](SONNET_FORENSIC_PROMPT.txt) — READ-ONLY audit prompt for Sonnet/Copilot
- [`VPS_FORENSIC_AUDIT_2026-03-03.md`](VPS_FORENSIC_AUDIT_2026-03-03.md) — Latest forensic audit (2026-03-03)

---

## 1. High-level Topology

```
[ Binance WS/REST ]  [ Bybit WS ]  [ OKX WS ]  [ CoinGecko ]  [ Funding Rates ]
         │                │               │              │               │
         └─────────────────┴───────────────┴──────────────┴───────────────┘
                                        │
                              ┌─────────▼──────────┐
                              │  L2: Data Ingestion │
                              │  market-publisher   │
                              │  cross-exchange     │
                              │  multi-source-feed  │
                              │  universe-os        │
                              └─────────┬──────────┘
                                        │ XADD quantum:stream:market.klines
                                        │
                         ┌──────────────▼───────────────┐
                         │  L3: AI Engine (Ensemble)     │
                         │  XGBoost + LightGBM           │
                         │  N-HiTS + PatchTST            │
                         │  model-federation             │
                         │  meta-regime                  │
                         │  CEO/Strategy/Risk brains     │
                         └──────────────┬───────────────┘
                                        │ XADD quantum:stream:trade.intent
                                        │
               ┌────────────────────────▼───────────────────────┐
               │  L4: Strategy & Decision                        │
               │  L5: Risk & Governance Gates                    │
               │  (exposure-balancer, portfolio-governance, etc) │
               └────────────────────────┬───────────────────────┘
                                        │ approved intents
                                        │
                         ┌──────────────▼────────────────┐
                         │  L6: Execution & Exits        │
                         │  execution-service            │
                         │  autonomous-trader            │
                         │  exitbrain-v3.5 (TP/SL)      │
                         │  exit-monitor (watchdog)      │
                         └──────────────┬────────────────┘
                                        │ orders
                                        ▼
                              [ Binance REST API ]

        ┌──────────────────────────────────────────────────────────────┐
        │                  REDIS EVENT BUS (6379)                      │
        │  Streams: market.klines, trade.intent, trade.closed, ...     │
        │  Keys:    governance:policy, rl:policy:*, regime:*, ...      │
        └──────────────────────────────────────────────────────────────┘
                         │            │            │
               ┌─────────┘   ┌────────┘   ┌───────┘
               │             │            │
      ┌────────▼──┐   ┌──────▼───┐   ┌────▼──────────────┐
      │ L7:       │   │ L8: RL   │   │ L1: UI/Dashboards │
      │ Analytics │   │ Training │   │ NGINX (443)       │
      │ PnL etc.  │   │ Monitor  │   │ Grafana (3000)    │
      └───────────┘   └──────────┘   └───────────────────┘
```

---

## 2. Public Entrypoints (Edge)

| Entrypoint | Binding | Description |
|-----------|---------|-------------|
| NGINX :80 | 0.0.0.0 | HTTP → HTTPS redirect |
| NGINX :443 | 0.0.0.0 | TLS termination, reverse proxy, rate limit (10 req/s) |
| SSH :22 | 0.0.0.0 | Admin access only |

### NGINX routes (active — `app.quantumfond.com`):

| Location | Upstream | Description |
|----------|---------|-------------|
| `/` | static HTML | QuantumFond frontend |
| `/api/` | `http://127.0.0.1:8000` | Backend API (dashboard) |
| `/api/rl-dashboard/` | `http://127.0.0.1:8000` | RL Dashboard via backend |
| `/grafana/api/quantum/` | `http://127.0.0.1:8000/grafana/api/quantum/` | Grafana quantum API bridge |
| `/grafana/` | `http://127.0.0.1:3000/grafana/` | Grafana dashboards |

---

## 3. Internal Services (full port map)

| Port | Binding | Service | Layer | Status |
|------|---------|---------|-------|--------|
| 6379 | 127.0.0.1 | Redis | Bus | ✅ |
| 8000 | 0.0.0.0 | Backend dashboard (run.py) | L1 | ✅ |
| 8001 | 127.0.0.1 | AI Engine (uvicorn) | L3 | ✅ |
| 8002 | 0.0.0.0 | Execution service | L6 | ✅ |
| 8004 | 127.0.0.1 | Portfolio Intelligence | L7 | ✅ |
| 8007 | 0.0.0.0 | Exit Monitor | L6 | ✅ |
| 8010 | 127.0.0.1 | CEO Brain | L3 | ✅ |
| 8011 | 127.0.0.1 | Strategy Brain | L3 | ✅ |
| 8012 | 127.0.0.1 | Risk Brain | L3 | ✅ |
| 3000 | * | Grafana | Observability | ✅ |
| 3100 | * | Loki | Observability | ✅ |
| 9091 | * | Prometheus | Observability | ✅ |
| 9100 | * | Node Exporter | Observability | ✅ |
| 9121 | * | Redis Exporter | Observability | ✅ |

---

## 4. Layers (VPS)

### L0 — Edge (NGINX)
- TLS termination (Let's Encrypt)
- Rate limiting: 10 req/s API, 1 req/s health
- Security headers: HSTS, X-Frame, X-Content-Type
- Routing to all internal services
- Config: `/etc/nginx/sites-available/`, `/etc/nginx/sites-enabled/`

### L1 — UI & API
- **Backend API** (port 8000): FastAPI — feeds dashboard, metrics
- **Grafana** (port 3000): Main monitoring UI (accessible via `/grafana/`)
- **QuantumFond Frontend**: Static HTML served by nginx

### L2 — Data Ingestion
| Service | Description | Output stream |
|---------|-------------|---------------|
| `quantum-market-publisher` | Binance WS — 30 symbols klines | `quantum:stream:market.klines` |
| `quantum-cross-exchange-aggregator` | Multi-exchange normalize & merge | `quantum:stream:exchange.normalized` |
| `quantum-multi-source-feed` | Bybit + OKX + CoinGecko + Fear&Greed + Funding Rates | `quantum:stream:market_events` |
| `quantum-universe-service` | Dynamic symbol universe management | — |
| `quantum-price-feed` | WebSocket → Redis price feed | `quantum:stream:exchange.raw` |

### L3 — AI Engine & Models
| Service | Port | Description |
|---------|------|-------------|
| `quantum-ai-engine` | 8001 | Ensemble inference, signal generation, feature engineering |
| `quantum-ceo-brain` | 8010 | AI orchestration — overall market strategy |
| `quantum-strategy-brain` | 8011 | Strategy evaluation per symbol |
| `quantum-risk-brain` | 8012 | Position sizing and risk assessment |
| `quantum-meta-regime` | — | Market regime classification |
| `quantum-model-federation` | — | Multi-model consensus + voting layer |
| `quantum-model-supervisor` (8007) | 8007 | Model drift detection, governance |
| `quantum-ensemble-predictor` | — | Shadow mode ensemble predictions |

**Ensemble models:** XGBoost · LightGBM · N-HiTS · PatchTST  
**Min. signal confidence:** 0.45  
**Output:** `quantum:stream:trade.intent` (via signal → decision pipeline)

### L4 — Strategy & Decision
| Service | Description |
|---------|-------------|
| `quantum-p35-decision-intelligence` | P3.5 decision engine — signal + regime + risk → intent |
| `quantum-autonomous-trader` | Full RL autonomy mode — scans + decides + exits |
| `quantum-ai-strategy-router` | Routes signals to appropriate strategy |
| `quantum-capital-allocation` | Kelly/RL capital distribution per symbol |
| `quantum-strategic-memory` | Learns trade patterns, cycle: 60s |
| `quantum-strategic-evolution` | Auto-evolves strategies based on performance |
| `quantum-scaling-orchestrator` | Adaptive sizing orchestrator (3C) |
| `quantum-harvest-brain` / `harvest-v2` | Profit harvesting brain |
| `quantum-governor` | P3.2 governance control |

### L5 — Risk & Governance
| Service | Description | Key state |
|---------|-------------|-----------|
| `quantum-risk-safety` | Hard limits: max leverage 10x, max pos $5k, max portfolio $25k | — |
| `quantum-exposure_balancer` | Margin util ≤85%, symbol ≤15%, min. 5 symbols | — |
| `quantum-portfolio-governance` | Dynamic policy based on historical performance | `quantum:governance:policy` |
| `quantum-risk-kernel` | Core risk: VaR, drawdown, correlation | `quantum:stream:risk.events` |
| `quantum-risk-guard` | Hard guardrails — blocks on violation | — |
| `quantum-portfolio-gate` | Portfolio allow-listing | `quantum:stream:portfolio.gate` |
| `quantum-portfolio-heat-gate` | Heat-based position gate (P2.6) | `quantum:stream:apply.heat.observed` |
| `quantum-portfolio-governance` | Exposure policy control | `quantum:governance:policy` |
| `quantum-dag3-hw-stops` | Hardware TP/SL guardian (DAG3) | — |
| `quantum-dag4-deadlock-guard` | Redis deadlock guard (XAUTOCLAIM) | — |
| `quantum-dag5-lockdown-guard` | DAG5 lockdown protection | — |
| `quantum-dag8-freeze-exit` | Freeze-exit analyzer & phased recovery | — |

### L6 — Execution & Exits (CRITICAL PATH)
| Service | Description |
|---------|-------------|
| `quantum-execution` | Order gateway → Binance REST API |
| `quantum-autonomous-trader` | Execution loop (also in L4) |
| `quantum-intent-bridge` | `trade.intent` → `apply.plan` bridge |
| `quantum-intent-executor` | P3.3 → Binance order placement |
| `quantum-position-monitor` | Position monitoring (no longer separate — embedded) |
| `quantum-exitbrain-v35` | ⚠️ ExitBrain v3.5 — intelligent TP/SL/trailing exits |
| `quantum-exit-monitor` | Watchdog: heartbeat monitoring → triggers `system:panic_close` |
| `quantum-exit-intelligence` | Exit intelligence service |
| `quantum-emergency-exit-worker` | Consumer of `system:panic_close` → closes all positions |
| `quantum-reconcile-engine` | P3.4 position reconciliation |

**Critical stream:** `system:panic_close` — if exit heartbeat lost, watchdog writes here → all positions closed unconditionally.

### L7 — Analytics
| Service | Port | Description |
|---------|------|-------------|
| `quantum-portfolio-intelligence` | 8004 | Real-time PnL, exposure per symbol |
| `quantum-pnl-reconciler` | — | Live Binance reward bridge |
| `quantum-performance-attribution` | — | P3.0 profit/loss attribution to strategies |
| `quantum-performance-tracker` | — | Performance tracking |
| `quantum-balance-tracker` | — | Binance account balance monitor |
| `quantum-trade-logger` | — | Trade history logger |
| `quantum-training-worker` | — | ML model auto-retraining (oneshot) |
| `quantum-harvest-metrics-exporter` | — | P2.7 harvest metrics |
| `quantum-layer6-post-trade` | — | Post-trade analytics |
| `quantum-runtime-truth` | — | LAG-1 observability engine |

### L8 — Reinforcement Learning
| Service | Description |
|---------|-------------|
| `quantum-rl-agent` | Continuous learning position sizing |
| `quantum-rl-trainer` | RL policy training consumer |
| `quantum-rl-sizer` | RL position sizing model server |
| `quantum-rl-feedback-v2` | Policy adjustment learning (feedback bridge v2) |
| `quantum-rl-monitor` | RL performance monitor + Discord alerts |
| `quantum-rl-shadow-metrics-exporter` | Shadow metrics for Prometheus |
| `quantum-rl-calibrator` | Calibrates RL weights vs. market performance |

---

## 5. Canonical Redis Contracts (minimum)

| Stream | Produced by | Consumed by | Direction |
|--------|------------|-------------|-----------|
| `quantum:stream:market.klines` | market-publisher | ai-engine | L2 → L3 |
| `quantum:stream:trade.intent` | ai-engine | execution, intent-bridge | L3 → L6 |
| `quantum:stream:trade.execution.res` | execution-service | autonomous-trader | L6 → L4 |
| `quantum:stream:trade.closed` | execution-service | analytics, RL | L6 → L7/L8 |
| `quantum:stream:portfolio.state` | portfolio-intelligence | governance, UI | L7 → L5/L1 |
| `system:panic_close` | risk-kernel / exit-monitor / ops | emergency-exit-worker | L5/L6 → L6 |
| `quantum:stream:rl_rewards` | pnl-reconciler | rl-trainer | L7 → L8 |
| `quantum:stream:sizing.decided` | rl-sizer / capital-allocation | execution | L4 → L6 |

---

## 6. Fail-Closed Safety Model

| Failure | Consequence | Mechanism |
|---------|------------|-----------|
| AI Engine fails | No signal → no trade | Signal required to proceed |
| Risk/Governance fails | Trade blocked at gate | Fail-closed gate (deny by default) |
| ExitBrain heartbeat missing | Watchdog triggers `system:panic_close` | exit-monitor → `system:panic_close` stream |
| Redis fails | System cannot coordinate → halt | All services blocked on Redis calls |
| Execution consumer stuck | Intents pile up, no orders placed | Consumer group lag → monitor |

---

## 7. "Good" State Checklist

```bash
# 1. NGINX up
systemctl status nginx

# 2. Redis alive
redis-cli ping  # → PONG

# 3. Streams grow
redis-cli XLEN quantum:stream:market.klines   # → growing
redis-cli XLEN quantum:stream:trade.intent    # → growing

# 4. Consumer groups not lagging
redis-cli XINFO GROUPS quantum:stream:trade.intent
# → lag should be 0 or small

# 5. No panic events
redis-cli XLEN system:panic_close   # → 0

# 6. ExitBrain heartbeat fresh
redis-cli XREVRANGE quantum:stream:exit_brain:heartbeat + - COUNT 1
# → timestamp within last 2 seconds

# 7. Critical services running
systemctl is-active quantum-ai-engine quantum-execution quantum-exitbrain-v35 quantum-exit-monitor
# → all: active

# 8. No crash loops
systemctl list-units "quantum-*.service" --all | grep -v "active\|dead\|exited"
```

---

## 8. Observability Stack

| Tool | Port | URL | Purpose |
|------|------|-----|---------|
| Grafana | 3000 | `https://app.quantumfond.com/grafana/` | Dashboards, alerts |
| Prometheus | 9091 | internal | Metrics scraping |
| Loki | 3100 | internal | Log aggregation |
| Promtail | 9080 | internal | Log shipping |
| Node Exporter | 9100 | internal | System metrics |
| Redis Exporter | 9121 | internal | Redis metrics |

---

## 9. Known Issues (as of 2026-03-03 audit)

| # | Issue | Layer | Severity |
|---|-------|-------|----------|
| 1 | SL/TP math bug — SELL side produces $96M stop_loss, negative TP | L3 | 🚨 CRITICAL |
| 2 | Execution consumer group stuck since ~Feb 16 (10,004 unprocessed intents) | L6 | 🚨 CRITICAL |
| 3 | ExitBrain v3.5 DISABLED + broken drop-in syntax | L6 | 🚨 CRITICAL |
| 4 | position-state-brain crash-loop 6568× (missing Binance creds) | L6 | 🚨 HIGH |
| 5 | Redis restarted 19h ago (VPS uptime 42d — mismatch) | L0 | ⚠️ HIGH |
| 6 | Exposure balancer + BSC dead since Feb 21 (no margin guardrail) | L5 | ⚠️ HIGH |
| 7 | NGINX config pollution — 10 server_name conflicts from stale .backup files | L0 | ⚠️ MEDIUM |
| 8 | RL layer fully gated (rl_gate_pass=false on all signals) | L8 | ⚠️ MEDIUM |
| 9 | Market regime always "unknown" in signals | L3 | ⚠️ MEDIUM |
| 10 | 20+ undocumented python3 processes on public ports 8042–8069 | L0 | ⚠️ MEDIUM |

See [`VPS_FORENSIC_AUDIT_2026-03-03.md`](VPS_FORENSIC_AUDIT_2026-03-03.md) for full details and fix proposals.
