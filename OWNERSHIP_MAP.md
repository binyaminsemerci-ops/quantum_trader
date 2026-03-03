# OWNERSHIP_MAP.md — Quantum Trader (VPS) v3
**Scope:** VPS runtime (systemd + Redis). Not local dev.  
**Architecture:** Event-driven microservices OS  
**Event Bus:** Redis Streams (primary), Redis Keys (state)  
**Orchestrator:** systemd  
**Ingress:** NGINX 80/443 → internal services  
**Last verified:** 2026-03-03 (see `VPS_FORENSIC_AUDIT_2026-03-03.md`)

---

## A) SOURCE OF TRUTH (VPS)

### Runtime control
```
systemd units:  quantum-*.service
unit files:     /etc/systemd/system/quantum-*.service
logs:           journalctl -u <unit>
env files:      /etc/quantum/<service>.env   (REDACT secrets when sharing)
repo path:      /home/qt/quantum_trader      (user: qt)
venv:           /opt/quantum/venvs/ai-engine/bin/python3
```

### Network edge
```
nginx:          reverse proxy + TLS termination
active config:  /etc/nginx/sites-enabled/app.quantumfond.com
  /              → frontend (static HTML)
  /api/          → http://127.0.0.1:8000  (backend dashboard)
  /grafana/      → http://127.0.0.1:3000  (Grafana)
```

### Event bus & state
```
Redis:          127.0.0.1:6379  (NOT public)
  Streams:      quantum:stream:*  (async bus)
  Keys:         quantum:governance:*, quantum:rl:*, quantum:regime:*, quantum:metrics:*
```

---

## B) LAYERS & OWNERS

---

### L0 — Edge / Ingress (NGINX)
**systemd unit:** `nginx.service`  
**Owner:** infra/nginx  
**Config dir:** `/etc/nginx/`  
**Active config:** `/etc/nginx/sites-enabled/app.quantumfond.com`  

**Responsibilities:**
- TLS termination (Let's Encrypt / `/etc/letsencrypt/`)
- HTTP → HTTPS redirect
- Rate limiting (10 req/s API)
- Security headers (HSTS, X-Frame-Options, X-Content-Type-Options)
- Reverse proxy routing to internal services

**Failure symptoms:** UI/API unreachable; 502/504 errors  
**First checks:**
```bash
systemctl status nginx
nginx -t
tail -100 /var/log/nginx/error.log
curl -I https://app.quantumfond.com/
```

**⚠️ Known issue (2026-03-03):** 5 stale `.backup*` files in `/etc/nginx/sites-enabled/` causing 10 conflicting server_name warnings. Config is valid but noisy.

---

### L1 — UI & API
**systemd units:**
- `quantum-backend.service` (port 8000)
**External UI:** Grafana on port 3000, served via nginx `/grafana/`

**Owner:** frontend / backend-api team  
**Responsibilities:** UI rendering, dashboards, read-only views into Redis/metrics  
**Dependencies:** nginx routing; backend → Redis  

**First checks:**
```bash
curl -I https://app.quantumfond.com/
curl -s https://app.quantumfond.com/api/health || true
systemctl status quantum-backend
journalctl -u quantum-backend --since "2 hours ago" | tail -50
```

---

### L2 — Data Ingestion
**systemd units:**
- `quantum-market-publisher.service`
- `quantum-cross-exchange-aggregator.service`
- `quantum-multi-source-feed.service`
- `quantum-universe-service.service`
- `quantum-universe.service`
- `quantum-price-feed.service`

**Produces:**
| Stream | Content |
|--------|---------|
| `quantum:stream:market.klines` | OHLCV ticks from Binance WS (30 symbols) |
| `quantum:stream:exchange.raw` | Raw exchange data |
| `quantum:stream:exchange.normalized` | Multi-exchange merged data |
| `quantum:stream:market_events` | CoinGecko + Fear&Greed + Funding Rates |

**Failure symptoms:** Stale prices; AI engine "no data"; streams not growing  
**First checks:**
```bash
systemctl status quantum-market-publisher quantum-cross-exchange-aggregator
redis-cli XLEN quantum:stream:market.klines
redis-cli XREVRANGE quantum:stream:market.klines + - COUNT 1
```

---

### L3 — AI Engine & Models
**systemd units:**
- `quantum-ai-engine.service` (port 8001)
- `quantum-ceo-brain.service` (port 8010)
- `quantum-strategy-brain.service` (port 8011)
- `quantum-risk-brain.service` (port 8012)
- `quantum-meta-regime.service`
- `quantum-model-federation.service`
- `quantum-model-supervisor.service` (port 8007)
- `quantum-ensemble-predictor.service`

**Ensemble models:** XGBoost · LightGBM · N-HiTS · PatchTST  
**Min. confidence threshold:** 0.45  
**WorkingDirectory:** `/opt/quantum`  
**ExecStart:** `python -m uvicorn microservices.ai_engine.main:app --host 127.0.0.1 --port 8001`

**Consumes:** `quantum:stream:market.klines`  
**Produces:** `quantum:stream:trade.intent` (after signal → decision pipeline)

**Failure symptoms:** No signals; model load errors; drift warnings; `signal:generated` not growing  
**First checks:**
```bash
systemctl status quantum-ai-engine
curl -s http://127.0.0.1:8001/health
redis-cli XLEN quantum:stream:trade.intent
redis-cli XREVRANGE quantum:stream:trade.intent + - COUNT 1
journalctl -u quantum-ai-engine --since "30 minutes ago" | tail -50
```

**⚠️ Known bug (2026-03-03):** SELL-side SL/TP math error — stop_loss and take_profit values are inverted/wrong for SELL orders. Affects all SELL intents in stream.  
**File:** `/home/qt/quantum_trader/microservices/ai_engine/main.py`

---

### L4 — Strategy & Decision
**systemd units:**
- `quantum-p35-decision-intelligence.service`
- `quantum-autonomous-trader.service`
- `quantum-ai-strategy-router.service`
- `quantum-capital-allocation.service`
- `quantum-strategic-memory.service`
- `quantum-strategic-evolution.service`
- `quantum-scaling-orchestrator.service`
- `quantum-governor.service`
- `quantum-harvest-brain.service`
- `quantum-harvest-v2.service`
- `quantum-harvest-optimizer.service`
- `quantum-harvest-proposal.service`
- `quantum-shadow-mode-controller.service`

**Consumes:** `quantum:stream:trade.intent` (AI signals)  
**Produces:** refined intents → passed to L5 gates  

**State keys:**
- `quantum:governance:last_decision`

**Failure symptoms:** Signals exist in stream but no orders appear; "gated" forever  
**First checks:**
```bash
systemctl status quantum-autonomous-trader quantum-p35-decision-intelligence
journalctl -u quantum-autonomous-trader --since "30 minutes ago" | tail -50
redis-cli XLEN quantum:stream:trade.intent
```

**⚠️ Known issue (2026-03-03):** `rl_gate_pass=false`, `rl_weight_effective=0.0` on ALL signals — RL has 0% influence. Kill-switch in ai-engine.env or main.py.

---

### L5 — Risk & Governance (GATES)
**systemd units:**
- `quantum-risk-safety.service`
- `quantum-risk-kernel.service`
- `quantum-risk-guard.service`
- `quantum-risk-brake.service`
- `quantum-risk-proposal.service`
- `quantum-portfolio-governance.service`
- `quantum-exposure_balancer.service`
- `quantum-portfolio-gate.service`
- `quantum-portfolio-heat-gate.service`
- `quantum-portfolio-risk-governor.service`
- `quantum-anti-churn-guard.service`
- `quantum-dag3-hw-stops.service`
- `quantum-dag4-deadlock-guard.service`
- `quantum-dag5-lockdown-guard.service`
- `quantum-dag8-freeze-exit.service`
- `quantum-bsc.service`

**Authoritative state key:** `quantum:governance:policy`  
**Consumes:** trade intents + portfolio state  
**Produces:** approvals / blocks / `system:panic_close`  

**Hard limits (risk-safety):**
- Max leverage: 10x
- Max position: $5,000 USD
- Max portfolio: $25,000 USD

**Soft limits (exposure-balancer):**
- Max margin utilization: 85%
- Max single symbol exposure: 15%
- Min diversification: 5 symbols
- Rebalance interval: 10s

**Failure symptoms:** All trades blocked; or emergency close events; or limits not enforced  
**First checks:**
```bash
systemctl status quantum-portfolio-governance quantum-exposure_balancer
redis-cli GET quantum:governance:policy
redis-cli XLEN system:panic_close
journalctl -u quantum-risk-safety --since "2 hours ago" | tail -50
```

**⚠️ Known issue (2026-03-03):** `quantum-exposure_balancer` DEAD since Feb 21. `quantum-bsc` DEAD since Feb 21. Margin guardrails NOT enforced.

---

### L6 — Execution & Exits (CRITICAL PATH)
**systemd units:**
- `quantum-execution.service` — order gateway → Binance REST
- `quantum-autonomous-trader.service` — full execution loop
- `quantum-intent-bridge.service` — trade.intent → apply.plan bridge
- `quantum-intent-executor.service` — final order placement (P3.3)
- `quantum-exitbrain-v35.service` — ❌ DISABLED (as of 2026-03-03)
- `quantum-exit-monitor.service` — watchdog
- `quantum-exit-intelligence.service`
- `quantum-emergency-exit-worker.service` — consumer of system:panic_close
- `quantum-reconcile-engine.service`
- `quantum-execution-result-bridge.service`

**Unit file paths:**
- `/etc/systemd/system/quantum-execution.service`
- `/etc/systemd/system/quantum-exitbrain-v35.service`
- `/etc/systemd/system/quantum-exit-monitor.service`

**ExecStart details:**
```
quantum-execution:       python3 services/execution_service.py
quantum-autonomous-trader: python -u microservices/autonomous_trader/autonomous_trader.py
quantum-exitbrain-v35:   python3 microservices/position_monitor/main_exitbrain.py
quantum-exit-monitor:    python3 services/exit_monitor_service_v2.py
```

**Consumes:** `quantum:stream:trade.intent` → consumer group `quantum:group:execution:trade.intent`  
**Produces:** exchange orders + exit actions + `system:panic_close` (via watchdog)

**Consumer groups on `quantum:stream:trade.intent`:**
| Group | Consumers | Pending | Status |
|-------|-----------|---------|--------|
| `quantum:group:execution:trade.intent` | 1 | 0 | 🚨 STUCK since ~2026-02-16 |
| `quantum:group:intent_bridge` | 30 | 4 | ✅ lag=0 |

**Failure symptoms:**
- Intents pile up, no orders → consumer group stuck
- Positions open but never close → exitbrain dead/disabled
- Heartbeat missing → watchdog triggers panic_close

**First checks:**
```bash
# Consumer group lag
redis-cli XINFO GROUPS quantum:stream:trade.intent

# Last execution result
redis-cli XREVRANGE quantum:stream:trade.execution.res + - COUNT 2

# ExitBrain status
systemctl status quantum-exitbrain-v35
journalctl -u quantum-exitbrain-v35 --since "30 minutes ago"

# Exit monitor (watchdog)
systemctl status quantum-exit-monitor
redis-cli XREVRANGE quantum:stream:exit_brain:heartbeat + - COUNT 1

# Panic close stream
redis-cli XLEN system:panic_close
```

**⚠️ Known issues (2026-03-03):**
1. `quantum-exitbrain-v35`: DISABLED, broken drop-in `/etc/systemd/system/quantum-exitbrain-v35.service.d/shadow.conf`
2. Execution consumer group `quantum:group:execution:trade.intent` stuck since ~Feb 16 (10,004 unprocessed intents)
3. Last execution result in stream: 2026-02-09 (REJECTED)
4. `quantum-position-state-brain`: crash-loop 6568× — "Missing Binance credentials"

---

### L7 — Analytics
**systemd units:**
- `quantum-portfolio-intelligence.service` (port 8004)
- `quantum-pnl-reconciler.service`
- `quantum-performance-attribution.service`
- `quantum-performance-tracker.service`
- `quantum-balance-tracker.service`
- `quantum-trade-logger.service`
- `quantum-training-worker.service` (oneshot — dead after run)
- `quantum-layer6-post-trade.service`
- `quantum-runtime-truth.service`

**Produces:** `quantum:stream:portfolio.state`, `quantum:stream:portfolio.snapshot_updated`, `quantum:stream:portfolio.exposure_updated`  

**Failure symptoms:** Dashboards stale; PnL mismatch; portfolio state key stale  
**First checks:**
```bash
systemctl status quantum-portfolio-intelligence quantum-pnl-reconciler
redis-cli XLEN quantum:stream:portfolio.state
redis-cli XREVRANGE quantum:stream:portfolio.state + - COUNT 1
```

---

### L8 — Reinforcement Learning
**systemd units:**
- `quantum-rl-agent.service`
- `quantum-rl-trainer.service`
- `quantum-rl-sizer.service`
- `quantum-rl-feedback-v2.service`
- `quantum-rl-monitor.service`
- `quantum-rl-shadow-metrics-exporter.service`

**State keys:**
```
quantum:rl:state:*          per position-id RL state
quantum:rl:policy:*         per symbol policy (e.g. quantum:rl:policy:BNBUSDT)
quantum:rl:calibration:v1:* per symbol calibration
```

**Heartbeat keys:**
```
quantum:svc:rl_trainer:heartbeat
quantum:svc:rl_feedback_v2:heartbeat
```

**Failure symptoms:** RL weights frozen; no rewards published; training stopped; `rl_gate_pass=false` in all signals  
**First checks:**
```bash
systemctl status quantum-rl-agent quantum-rl-trainer
redis-cli XLEN quantum:stream:rl_rewards
redis-cli XREVRANGE quantum:stream:rl_rewards + - COUNT 1
redis-cli GET quantum:svc:rl_trainer:heartbeat
```

**⚠️ Known issue (2026-03-03):** RL gated out (`rl_gate_pass=false`) on all signals despite services running. Check `RL_ENABLED` flag in `/etc/quantum/ai-engine.env`.

---

## C) REDIS STREAM CONTRACTS (COMPLETE)

### Primary bus streams:

| Stream key | Produced by | Consumed by | Notes |
|------------|------------|-------------|-------|
| `quantum:stream:market.klines` | market-publisher | ai-engine (XREAD) | 30 symbols, Binance WS |
| `quantum:stream:exchange.raw` | price-feed | cross-exchange | raw ticks |
| `quantum:stream:exchange.normalized` | cross-exchange-aggregator | ai-engine, strategy | multi-exchange merge |
| `quantum:stream:market_events` | multi-source-feed | ai-engine | Bybit+OKX+CoinGecko |
| `quantum:stream:features` | feature-publisher | ai-engine | Engineered features |
| `quantum:stream:ai.signal_generated` | ai-engine | decision-intelligence | Pre-decision signals |
| `quantum:stream:ai.decision.made` | decision-intelligence | execution | Post-decision |
| `quantum:stream:trade.intent` | ai-engine, autonomous-trader | execution (×2 groups) | CENTRAL EXECUTION STREAM |
| `quantum:stream:trade.execution.res` | execution-service | analytics, autonomous-trader | Order results |
| `quantum:stream:trade.closed` | execution-service | analytics, RL | Closed position events |
| `quantum:stream:trade.signal.v5` | signal-injector / chaos | strategy | v5 signal format |
| `quantum:stream:portfolio.state` | portfolio-intelligence | governance, UI | Portfolio snapshot |
| `quantum:stream:portfolio.snapshot_updated` | portfolio-intelligence | multiple | Snapshot update events |
| `quantum:stream:portfolio.exposure_updated` | exposure-balancer | governance | Exposure changes |
| `quantum:stream:portfolio.cluster_state` | portfolio-clusters | governance | Cluster state |
| `quantum:stream:portfolio.gate` | portfolio-gate | execution | Gate decisions |
| `quantum:stream:risk.events` | risk-kernel | governance, watchdog | Risk alerts |
| `quantum:stream:apply.plan.manual` | manual / ops | apply-layer | Manual ops |
| `quantum:stream:apply.heat.observed` | heat-gate | apply-layer | Heat events |
| `quantum:stream:bsc.events` | bsc | governance | BSC safety events |
| `quantum:stream:reconcile.close` | reconcile-engine | execution | Reconciliation closes |
| `quantum:stream:reconcile.alert` | reconcile-engine | ops/monitoring | Reconciliation alerts |
| `quantum:stream:hover.intent` | harvest-brain | harvest-optimizer | Harvest intents |
| `quantum:stream:harvest.proposal` | harvest-proposal | harvest-brain | Harvest proposals |
| `quantum:stream:harvest.intent` | harvest-brain | execution | Harvest executions |
| `quantum:stream:rl_rewards` | pnl-reconciler | rl-trainer | RL reward signals |
| `quantum:stream:rl.stats` | rl-agent | rl-monitor | RL statistics |
| `quantum:stream:sizing.decided` | rl-sizer / capital-alloc | execution | Final sizing |
| `quantum:stream:policy.update` | portfolio-governance | strategy-ops | Policy changes |
| `quantum:stream:policy.updated` | portfolio-governance | multiple | Policy update events |
| `quantum:stream:policy.audit` | portfolio-governance | audit-logger | Policy audit trail |
| `quantum:stream:model.retrain` | strategic-evolution | training-worker | Retrain triggers |
| `quantum:stream:ai.exit.decision` | exitbrain-v3.5 | exit-monitor | Exit decisions |
| `system:panic_close` | risk-kernel / exit-monitor / ops | emergency-exit-worker | EMERGENCY ALL-CLOSE |
| `system:panic_close:completed` | emergency-exit-worker | audit-logger | EEW completion report |

### Key state (Redis HASH / STRING):

| Key pattern | Written by | Read by | Description |
|------------|-----------|---------|-------------|
| `quantum:governance:policy` | portfolio-governance | ai-engine, strategy-ops | Active policy document |
| `quantum:governance:last_decision` | portfolio-governance | monitoring | Last decision timestamp |
| `quantum:rl:state:<position_id>` | rl-agent | rl-trainer, autonomous-trader | Per-position RL state |
| `quantum:rl:policy:<symbol>` | rl-trainer | ai-engine, rl-sizer | Per-symbol RL policy |
| `quantum:rl:calibration:v1:<symbol>` | rl-calibrator | rl-sizer | Calibration weights |
| `quantum:regime:<symbol>` | meta-regime | ai-engine | Market regime class |
| `quantum:metrics:exit:*` | exit-intelligence | monitoring | Exit metrics (298k+ records) |
| `quantum:permit:p33:<id>` | intent-executor | position-state-brain | Trade permits |
| `quantum:intent_bridge:seen:<id>` | intent-bridge | intent-bridge | Dedup cache (TTL) |
| `quantum:intent_executor:done:<id>` | intent-executor | — | Completed intent tracking |
| `quantum:svc:rl_trainer:heartbeat` | rl-trainer | rl-monitor | Service heartbeat |
| `quantum:svc:rl_feedback_v2:heartbeat` | rl-feedback-v2 | rl-monitor | Service heartbeat |

---

## D) WHEN X BREAKS — WHERE TO LOOK

### 1. No UI / API (502/504)
```
nginx status? → upstream ports alive? → backend health?

systemctl status nginx
curl -s http://127.0.0.1:8000/health
journalctl -u nginx --since "30 minutes ago" | tail -30
```

### 2. No signals (AI silent)
```
Is market:data growing?
  YES → ai-engine problem
  NO  → ingestion problem

# Check ingestion:
redis-cli XLEN quantum:stream:market.klines
systemctl status quantum-market-publisher
journalctl -u quantum-market-publisher --since "30 min ago" | tail -30

# Check AI:
systemctl status quantum-ai-engine
curl -s http://127.0.0.1:8001/health
journalctl -u quantum-ai-engine --since "30 min ago" | tail -30
```

### 3. Signals exist, no intents
```
Decision/strategy layer blocking?

redis-cli XLEN quantum:stream:trade.intent
journalctl -u quantum-p35-decision-intelligence --since "30 min ago" | tail -30
journalctl -u quantum-autonomous-trader --since "30 min ago" | tail -30
redis-cli GET quantum:governance:policy
```

### 4. Intents exist, no orders placed
```
Execution consumer stuck?

redis-cli XINFO GROUPS quantum:stream:trade.intent
# Check last-delivered-id timestamp vs. current time
# If stuck: reset with XGROUP SETID ... '$' then restart execution

redis-cli XREVRANGE quantum:stream:trade.execution.res + - COUNT 2
journalctl -u quantum-execution --since "2 hours ago" | tail -50
```

### 5. Positions won't close
```
ExitBrain running? Heartbeat fresh?

systemctl status quantum-exitbrain-v35
redis-cli XREVRANGE quantum:stream:exit_brain:heartbeat + - COUNT 1
# Heartbeat must be within last 2 seconds

systemctl status quantum-exit-monitor
redis-cli XLEN system:panic_close
```

### 6. Emergency close triggered (panic_close > 0)
```
redis-cli XREVRANGE system:panic_close + - COUNT 3
# Check 'issued_by' and 'reason' fields

redis-cli XREVRANGE system:panic_close:completed + - COUNT 3
# Check positions_closed vs positions_total
```

### 7. RL not influencing trades
```
rl_gate_pass=false in trade.intent events?

cat /etc/quantum/ai-engine.env | grep -i rl   # (REDACT secrets when sharing)
grep -n "rl_gate\|RL_ENABLED\|rl_influence" /home/qt/quantum_trader/microservices/ai_engine/main.py
redis-cli KEYS "quantum:rl:policy:*" | wc -l
```

### 8. Market regime always "unknown"
```
meta-regime service writing to correct key?

systemctl status quantum-meta-regime
redis-cli KEYS "quantum:regime:*"
# If empty → meta-regime not writing, or key name mismatch with ai-engine reader
grep -n "regime" /home/qt/quantum_trader/microservices/meta_regime/main.py | head -20
grep -n "regime" /home/qt/quantum_trader/microservices/ai_engine/main.py | head -20
```

---

## E) ENVIRONMENT FILES (VPS)

Located at `/etc/quantum/` — **NEVER commit values, only names**

| File | Used by |
|------|---------|
| `ai-engine.env` | quantum-ai-engine, quantum-execution |
| `autonomous-trader.env` | quantum-autonomous-trader |
| `exitbrain-v35.env` | quantum-exitbrain-v35 |
| `exitbrain-control.env` | quantum-exitbrain-v35 (drop-in) |
| `testnet.env` | quantum-execution, quantum-paper-trade-controller |
| `intent-executor.env` | quantum-execution |
| `apply-layer.env` | quantum-apply-layer, quantum-position-state-brain |
| `position-state-brain.env` | quantum-position-state-brain (⚠️ MISSING Binance creds) |

**Variables expected (names only):**
- `BINANCE_API_KEY`
- `BINANCE_API_SECRET`
- `BINANCE_USE_TESTNET`
- `REDIS_HOST` / `REDIS_PORT` / `REDIS_URL`
- `DISCORD_WEBHOOK_URL`
- `RL_ENABLED` (suspected gate flag)

---

## F) SYSTEMD QUICK REFERENCE

```bash
# List all quantum units
systemctl list-units "quantum-*.service" --all --no-pager

# Check if any failed
systemctl --failed

# Status of specific service
systemctl status quantum-<service>

# Logs (last 2 hours)
journalctl -u quantum-<service> --since "2 hours ago" --no-pager | tail -200

# Restart a service (CAUTION - not read-only)
systemctl restart quantum-<service>

# See all unit files
find /etc/systemd/system -name "quantum-*.service" | wc -l
find /etc/systemd/system -name "quantum-*.service" | sort
```

---

*Last updated: 2026-03-03 | Source: VPS Forensic Audit*
