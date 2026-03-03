# Quantum Trader — VPS Forensic Audit Report
**Date:** 2026-03-03 01:32–01:37 UTC  
**Mode:** READ-ONLY  
**Auditor:** GitHub Copilot — SSH forensic session  
**VPS:** Hetzner `46.224.116.254`  

---

## 1. VPS IDENTITY

| Field | Value |
|-------|-------|
| Hostname | `quantumtrader-prod-1` |
| Date (UTC) | Tue Mar 3 01:32:46 AM UTC 2026 |
| Kernel | Linux 6.8.0-90-generic #91-Ubuntu SMP PREEMPT_DYNAMIC x86_64 |
| VPS Uptime | 42 days, 02:47 (since ~Jan 19 2026) |
| Load average | 3.28 / 2.66 / 2.16 ← HIGH |
| Login user | root |
| Repo path | `/home/qt/quantum_trader` (owner: qt) |
| Alt paths | `/opt/quantum` (venvs, model registry) |
| Primary venv | `/opt/quantum/venvs/ai-engine/` |
| Systemd units | 137 quantum-*.service files installed, 97 loaded |

---

## 2. SYSTEMD INVENTORY (2026-03-03)

**Command used:** `systemctl list-units 'quantum-*.service' --all --no-pager`

### State breakdown (97 loaded units):

| State | Count |
|-------|-------|
| `active (running)` | ~80 |
| `active (exited)` | 1 (quantum-layer1-historical-backfill — oneshot, expected) |
| `activating (auto-restart)` | 1 ← **CRASH LOOP** |
| `inactive (dead)` | ~14 |
| `not-found inactive` | 2 (quantum-ensemble, quantum-redis — deleted unit files referenced elsewhere) |
| **FAILED** | **0** (systemctl --failed returned 0 units) |

### Complete unit list (all 97 loaded):

```
quantum-ai-engine.service                  active  running
quantum-ai-strategy-router.service         active  running
quantum-anti-churn-guard.service           active  running
quantum-apply-layer.service                active  running
quantum-autonomous-trader.service          active  running
quantum-backend.service                    active  running
quantum-balance-tracker.service            active  running
quantum-bsc.service                        loaded  inactive (dead) since Feb 21
quantum-capital-allocation.service         active  running
quantum-ceo-brain.service                  active  running
quantum-core-health.service                loaded  inactive (dead)
quantum-cross-exchange-aggregator.service  active  running
quantum-dag3-hw-stops.service              active  running
quantum-dag4-deadlock-guard.service        active  running
quantum-dag5-lockdown-guard.service        active  running
quantum-dag8-freeze-exit.service           active  running
quantum-diagnostic.service                 loaded  inactive (dead)
quantum-emergency-exit-worker.service      active  running
quantum-ensemble-predictor.service         active  running
quantum-ensemble.service                   not-found inactive
quantum-ess-watch.service                  loaded  inactive (dead)
quantum-exchange-stream-bridge.service     active  running
quantum-execution-result-bridge.service    active  running
quantum-execution.service                  active  running
quantum-exit-intelligence.service          active  running
quantum-exit-monitor.service               active  running
quantum-exit-owner-watch.service           loaded  inactive (dead)
quantum-exposure_balancer.service          loaded  inactive (dead) since Feb 21
quantum-feature-publisher.service          active  running
quantum-governor.service                   active  running
quantum-harvest-brain-2.service            active  running
quantum-harvest-brain.service              active  running
quantum-harvest-metrics-exporter.service   active  running
quantum-harvest-optimizer.service          active  running
quantum-harvest-proposal.service           active  running
quantum-harvest-v2.service                 active  running
quantum-health-gate.service                loaded  inactive (dead)
quantum-heat-gate.service                  active  running
quantum-intent-bridge.service              active  running
quantum-intent-executor.service            active  running
quantum-layer1-data-sink.service           active  running
quantum-layer1-historical-backfill.service active  exited (oneshot complete)
quantum-layer2-research-sandbox.service    active  running
quantum-layer2-signal-promoter.service     active  running
quantum-layer3-backtest-runner.service     active  running
quantum-layer4-portfolio-optimizer.service loaded  inactive (dead)
quantum-layer5-execution-monitor.service   active  running
quantum-layer6-post-trade.service          active  running
quantum-ledger-sync.service                loaded  inactive (dead)
quantum-market-publisher.service           active  running
quantum-marketstate.service                active  running
quantum-meta-regime.service                active  running
quantum-metricpack-builder.service         active  running
quantum-multi-source-feed.service          active  running
quantum-p28a3-latency-proof.service        loaded  inactive (dead)
quantum-p35-decision-intelligence.service  active  running
quantum-paper-trade-controller.service     active  running
quantum-performance-attribution.service    active  running
quantum-performance-tracker.service        active  running
quantum-pnl-reconciler.service             active  running
quantum-policy-sync.service                loaded  inactive (dead)
quantum-portfolio-clusters.service         active  running
quantum-portfolio-gate.service             active  running
quantum-portfolio-governance.service       active  running
quantum-portfolio-heat-gate.service        active  running
quantum-portfolio-intelligence.service     active  running
quantum-portfolio-risk-governor.service    active  running
quantum-portfolio-state-publisher.service  active  running
quantum-position-state-brain.service       activating  auto-restart  ← CRASH LOOP
quantum-price-feed.service                 active  running
quantum-reconcile-engine.service           active  running
quantum-redis.service                      not-found inactive
quantum-risk-brain.service                 active  running
quantum-risk-brake.service                 active  running
quantum-risk-proposal.service              active  running
quantum-risk-safety.service                active  running
quantum-rl-agent.service                   active  running
quantum-rl-feedback-v2.service             active  running
quantum-rl-monitor.service                 active  running
quantum-rl-shadow-metrics-exporter.service active  running
quantum-rl-shadow-scorecard.service        loaded  inactive (dead)
quantum-rl-sizer.service                   active  running
quantum-rl-trainer.service                 active  running
quantum-runtime-truth.service              active  running
quantum-safety-telemetry.service           active  running
quantum-scaling-orchestrator.service       active  running
quantum-shadow-mode-controller.service     active  running
quantum-signal-injector.service            active  running
quantum-strategic-memory.service           active  running
quantum-strategy-brain.service             active  running
quantum-stream-bridge.service              active  running
quantum-stream-recover.service             loaded  inactive (dead)
quantum-trade-logger.service               active  running
quantum-training-worker.service            loaded  inactive (dead)
quantum-universe-service.service           active  running
quantum-universe.service                   active  running
quantum-utf-publisher.service              active  running
```

### Critical service details (ExecStart / User / EnvironmentFile):

| Unit | Active | ExecStart | User | EnvironmentFile(s) | Port |
|------|--------|-----------|------|---------------------|------|
| quantum-ai-engine | ✅ running | uvicorn microservices.ai_engine.main:app --host 127.0.0.1 --port 8001 | qt | /etc/quantum/ai-engine.env | 8001 |
| quantum-autonomous-trader | ✅ running | python -u microservices/autonomous_trader/autonomous_trader.py | qt | /etc/quantum/autonomous-trader.env | — |
| quantum-execution | ✅ running | python3 services/execution_service.py | qt | ai-engine.env + testnet.env + intent-executor.env | 8002 |
| quantum-exit-monitor | ✅ running | python3 services/exit_monitor_service_v2.py | qt | — | 8007 |
| quantum-exitbrain-v35 | ❌ DISABLED/dead | python3 microservices/position_monitor/main_exitbrain.py | qt | /etc/quantum/exitbrain-v35.env | — |
| quantum-position-state-brain | 🔴 crash-loop #6568 | /usr/bin/python3 microservices/position_state_brain/main.py | **root** | apply-layer.env + position-state-brain.env | — |
| quantum-exposure_balancer | ❌ dead since Feb 21 | — | — | — | — |
| quantum-bsc | ❌ dead since Feb 21 | — | — | — | — |
| quantum-backend | ✅ running | python run.py | qt | — | 8000 |
| quantum-ceo-brain | ✅ running | python -m uvicorn backend.ai_orchestrator.service:app ... | qt | ai-engine.env | 8010 |
| quantum-strategy-brain | ✅ running | uvicorn backend.strategy_brain.service:app ... | qt | — | 8011 |
| quantum-risk-brain | ✅ running | uvicorn backend.risk_brain.service:app ... | qt | — | 8012 |

---

## 3. NGINX ROUTING PROOF

**nginx -t result:**
```
nginx: [warn] conflicting server name "app.quantumfond.com" on 0.0.0.0:443, ignored (×5)
nginx: [warn] conflicting server name "app.quantumfond.com" on 0.0.0.0:80, ignored  (×5)
nginx: the configuration file /etc/nginx/nginx.conf syntax is ok
nginx: configuration file /etc/nginx/nginx.conf test is successful
```

### Files in `/etc/nginx/sites-enabled/`:
```
app.quantumfond.com            ← ACTIVE (Jan 16 2026, current)
app.quantumfond.com.backup     ← stale, still parsed by nginx
app.quantumfond.com.backup2    ← stale, still parsed
app.quantumfond.com.backup3    ← stale, still parsed
app.quantumfond.com.backup_jan16 ← stale, still parsed
app.quantumfond.com.bak        ← stale, still parsed
investor.quantumfond.com       ← symlink → sites-available/
quantum.quantumfond.com        ← symlink → sites-available/
rl.quantumfond.com             ← symlink → sites-available/
```

### Active routing (from `app.quantumfond.com`):
```nginx
server_name  app.quantumfond.com;
listen 443 ssl;
listen 80;

# Stubs (return 204)
location = /grafana/api/user/stars         → 204
location = /log-grafana-javascript-agent   → 204

# API routing
location /grafana/api/quantum/  → proxy_pass http://127.0.0.1:8000/grafana/api/quantum/
location /grafana/              → proxy_pass http://127.0.0.1:3000/grafana/
location /api/rl-dashboard/     → proxy_pass http://127.0.0.1:8000
location /api/                  → proxy_pass http://127.0.0.1:8000
location /                      → static files (QuantumFond frontend HTML)
```

> **⚠️ Note:** Port 3000 = Grafana (not Next.js as per old design). The nginx correctly routes `/grafana/` to Grafana.

---

## 4. OPEN PORTS (ss -lntp snapshot 2026-03-03)

| Port | Bind | Process | Expected/Notes |
|------|------|---------|----------------|
| 22 | 0.0.0.0 + [::] | sshd | ✅ |
| 53 | 127.0.0.53 | systemd-resolved | ✅ |
| 80 | 0.0.0.0 + [::] | nginx (5 workers) | ✅ |
| 443 | 0.0.0.0 + [::] | nginx (5 workers) | ✅ |
| **3000** | `*` | **grafana** (pid=2333305) | ⚠️ Old design expected Next.js |
| 3100 | `*` | loki | ✅ observability |
| **6379** | **127.0.0.1 only** | redis-server (pid=1732045) | ✅ CORRECT — not public |
| 8000 | 0.0.0.0 | python run.py — backend dashboard (pid=1732055) | ✅ |
| **8001** | **127.0.0.1 only** | python uvicorn ai-engine (pid=1732052) | ✅ CORRECT |
| **8002** | **0.0.0.0** | python3 execution_service.py (pid=1732234) | ⚠️ publicly exposed |
| 8004 | 127.0.0.1 | python portfolio-intelligence (pid=1732072) | ✅ |
| 8005 | 0.0.0.0 | python3 unknown | ⚠️ identify |
| 8007 | 0.0.0.0 | python3 exit-monitor (pid=1732235) | ⚠️ publicly exposed |
| 8010 | 127.0.0.1 | python CEO-brain (pid=1731283) | ✅ |
| 8011 | 127.0.0.1 | python Strategy-brain (pid=1731307) | ✅ |
| 8012 | 127.0.0.1 | python Risk-brain (pid=1731292) | ✅ |
| 8020 | 0.0.0.0 | python3 unknown | ⚠️ identify |
| 8042 | 0.0.0.0 | python3 pid=2602220 | ⚠️ undocumented |
| 8043 | 0.0.0.0 | python3 pid=1732106 | ⚠️ undocumented |
| 8044 | 0.0.0.0 | python3 pid=1732061 | ⚠️ undocumented |
| 8046 | 0.0.0.0 | python3 pid=1374674 | ⚠️ undocumented |
| 8047 | 0.0.0.0 | python3 pid=2602270 | ⚠️ undocumented |
| 8048 | 0.0.0.0 | python3 pid=2602252 | ⚠️ undocumented |
| 8049 | 0.0.0.0 | python pid=1732073 | ⚠️ undocumented |
| 8051 | 0.0.0.0 | python3 pid=1732069 | ⚠️ undocumented |
| 8052 | 0.0.0.0 | python3 pid=1732062 | ⚠️ undocumented |
| 8056 | 0.0.0.0 | python3 pid=2602330 | ⚠️ undocumented |
| 8059 | 0.0.0.0 | python3 pid=2602191 | ⚠️ undocumented |
| 8061 | 0.0.0.0 | python3 pid=2602269 | ⚠️ undocumented |
| 8068 | 0.0.0.0 | python3 pid=2602213 | ⚠️ undocumented |
| 8069 | 0.0.0.0 | python3 pid=2602213 | ⚠️ undocumented |
| 9080 | `*` | promtail | ✅ |
| 9091 | `*` | prometheus | ✅ |
| 9092 | 0.0.0.0 | python3 pid=1732083 | ⚠️ undocumented |
| 9095 | `*` | loki | ✅ |
| 9100 | `*` | node-exporter | ✅ |
| 9109 | `*` | ? | ⚠️ |
| 9121 | `*` | prometheus-redis-exporter | ✅ |
| 44015 | `*` | promtail | ✅ |
| **3002** | — | **NOT LISTENING** | ❌ Design expected: QuantumFond frontend |
| **8006** | — | **NOT LISTENING** | ❌ Design expected: Universe OS API |
| **8026** | — | **NOT LISTENING** | ❌ Design expected: RL Dashboard |
| **8888** | — | **NOT LISTENING** | ❌ Design expected: nginx RL proxy |

---

## 5. REDIS STATUS

**Command:** `redis-cli ping`, `redis-cli INFO server`, stream probes

```
PING          : PONG
redis_version : 7.0.15
bind          : 127.0.0.1 (not public — correct)
config_file   : /etc/redis/redis.conf
hz            : 10
uptime_seconds: 68,393 (~19 hours)  ← MISMATCH: VPS uptime=42 days
                                       Redis restarted ~2026-03-02 06:37 UTC
connected_clients : 134
blocked_clients   : 30  (XREAD blocking—normal)
total_keys        : 298,029
```

### Active Redis Streams (confirmed as type=stream):

| Stream key (actual) | XLEN | Most recent event | Source | Status |
|---------------------|------|-------------------|--------|--------|
| `quantum:stream:trade.intent` | 10,004 | 2026-03-03 01:26:50 UTC | ai-engine | ✅ ACTIVE |
| `quantum:stream:trade.execution.res` | 2,154 | **2026-02-09 23:20 UTC** | execution-service | ❌ 23 days stale |
| `quantum:stream:trade.closed` | 1,006 | — | — | present |
| `quantum:stream:trade.signal.v5` | 500 | seq 499, `source: chaos_load_test` | chaos_load_test | ⚠️ test data only |
| `quantum:stream:ai.signal_generated` | — | — | — | present |
| `quantum:stream:ai.exit.decision` | — | — | — | present |
| `quantum:stream:ai.decision.made` | — | — | — | present |
| `quantum:stream:market.klines` | — | — | market-publisher | ✅ active (AI reads it) |
| `quantum:stream:exchange.raw` | — | — | — | present |
| `quantum:stream:exchange.normalized` | — | — | — | present |
| `quantum:stream:features` | — | — | — | present |
| `quantum:stream:portfolio.state` | — | — | — | present |
| `quantum:stream:portfolio.snapshot_updated` | — | — | — | present |
| `quantum:stream:portfolio.exposure_updated` | — | — | — | present |
| `quantum:stream:portfolio.cluster_state` | — | — | — | present |
| `quantum:stream:portfolio.gate` | — | — | — | present |
| `quantum:stream:harvest.proposal` | — | — | — | present |
| `quantum:stream:harvest.intent` | — | — | — | present |
| `quantum:stream:rl_rewards` | — | — | — | present |
| `quantum:stream:rl.stats` | — | — | — | present |
| `quantum:stream:sizing.decided` | — | — | — | present |
| `quantum:stream:policy.update` | — | — | — | present |
| `quantum:stream:policy.updated` | — | — | — | present |
| `quantum:stream:policy.audit` | — | — | — | present |
| `quantum:stream:model.retrain` | — | — | — | present |
| `quantum:stream:risk.events` | — | — | — | present |
| `quantum:stream:bsc.events` | — | — | — | present |
| `quantum:stream:reconcile.close` | — | — | — | present |
| `quantum:stream:reconcile.alert` | — | — | — | present |
| `quantum:stream:market_events` | — | — | — | present |
| `quantum:stream:position.snapshot` | — | — | — | present |
| `quantum:stream:apply.plan.manual` | — | — | — | present |
| `quantum:stream:apply.heat.observed` | — | — | — | present |
| `system:panic_close` | **0** | — | — | EXISTS, no events (correct) |

### Redis consumer groups:

| Stream | Group name | Consumers | Pending | Last Delivered ID | Lag |
|--------|-----------|-----------|---------|-------------------|-----|
| `quantum:stream:trade.intent` | `quantum:group:execution:trade.intent` | 1 | 0 | `1771160145484-0` (~Feb 16) | **NULL** 🚨 |
| `quantum:stream:trade.intent` | `quantum:group:intent_bridge` | 30 | 4 | `1772501750890-0` (Mar 3) | **0** ✅ |

### Active Redis state keys (confirmed):

| Key pattern | Purpose |
|------------|---------|
| `quantum:governance:policy` | Active portfolio policy |
| `quantum:governance:last_decision` | Last governance decision |
| `quantum:rl:state:*` | RL agent state (per position ID) |
| `quantum:rl:policy:*` | RL policy (per symbol, e.g. BNBUSDT) |
| `quantum:rl:calibration:v1:*` | RL calibration per symbol |
| `quantum:metrics:exit:*` | Exit metrics (298k+ records) |
| `quantum:permit:p33:*` | P3.3 trade permits |
| `quantum:intent_bridge:seen:*` | Intent dedup cache |
| `quantum:intent_executor:done:*` | Completed intent tracking |
| `quantum:svc:rl_trainer:heartbeat` | RL trainer heartbeat |
| `quantum:svc:rl_feedback_v2:heartbeat` | RL feedback heartbeat |

---

## 6. SERVICE-TO-STREAM OWNERSHIP TABLE

| Stream (actual key) | Producer | Consumer group | Consumer service | Lag |
|---------------------|---------|----------------|-----------------|-----|
| `quantum:stream:trade.intent` | ai-engine | `quantum:group:execution:trade.intent` | quantum-execution | 🚨 STUCK ~Feb 16 |
| `quantum:stream:trade.intent` | ai-engine | `quantum:group:intent_bridge` | quantum-intent-bridge (30 consumers) | ✅ 0 |
| `quantum:stream:trade.execution.res` | execution-service | none registered | autonomous-trader? | ❌ last msg Feb 9 |
| `quantum:stream:trade.signal.v5` | chaos_load_test | UNKNOWN | UNKNOWN | stale (test data) |
| `quantum:stream:ai.signal_generated` | ai-engine (implied) | UNKNOWN | UNKNOWN | — |
| `quantum:stream:market.klines` | market-publisher | direct XREAD | ai-engine | ✅ active |
| `quantum:stream:rl_rewards` | pnl-reconciler | UNKNOWN | rl-trainer | — |
| `quantum:stream:policy.update` | portfolio-governance | UNKNOWN | strategy-ops? | — |
| `system:panic_close` | risk-kernel / ops | `emergency_exit_worker` | quantum-emergency-exit-worker | 0 events ✅ |

---

## 7. HEALTH VERDICT PER LAYER

| Layer | Services | Verdict | Key evidence |
|-------|---------|---------|-------------|
| **L0 NGINX** | nginx reverse proxy | ⚠️ DEGRADED | 10 server_name conflicts from stale .backup files in sites-enabled. Config passes but with warnings. |
| **L1 UI** | QuantumFond frontend, dashboard, RL UI | ⚠️ PARTIAL | Port 8000 backend ✅. Port 3000 = Grafana (not Next.js). Ports 3002, 8026 NOT listening. Static HTML served via nginx `/`. |
| **L2 Data Ingestion** | market-publisher, cross-exchange, multi-source, universe | ✅ WORKS | quantum-market-publisher active. market.klines stream populated. AI engine reading it (confirmed in logs). |
| **L3 AI Engine** | ai-engine (8001), CEO/strategy/risk brain, meta-regime, model-federation | ✅ WORKS* | PID 1732052 active 482MB RAM. Generating live signals for BTC, APT, BNB. **SL/TP calculation bug on SELL side** (see breakpoint #1). |
| **L4 Risk/Governance** | risk-safety, exposure_balancer, portfolio-governance, risk-kernel, bsc | ⚠️ DEGRADED | portfolio-governance ✅ active, `quantum:governance:policy` key exists. **exposure_balancer DEAD** since Feb 21. **bsc DEAD** since Feb 21. No margin exposure guardrail. |
| **L5 Strategy/Decision** | autonomous-trader, decision-intelligence, capital-allocation, strategic-memory, strategic-evolution | ✅ MOSTLY WORKS | autonomous-trader completing cycles (0 entries, 9 exits per cycle log). p35-decision-intelligence active. RL gated out (rl_gate_pass=false). |
| **L6 EXECUTION** | execution, trade-intent-consumer, position-monitor, exitbrain-v35, exit-monitor | 🚨 BROKEN | Last execution result: Feb 9 (REJECTED). Execution consumer group stuck since ~Feb 16. ExitBrain v3.5 DISABLED. position-state-brain crash-looping 6568×. |
| **L7 Analytics** | portfolio-intelligence, PIL, pnl-tracker, retraining-worker, performance-attribution | ⚠️ DEGRADED | portfolio-intelligence ✅, pnl-reconciler ✅, performance-attribution ✅. **training-worker INACTIVE** (oneshot, dead). |
| **L8 RL** | rl-agent, rl-trainer, rl-calibrator, rl-monitor, rl-feedback-v2 | ⚠️ PRESENT/GATED | All services running. `quantum:rl:policy:*` keys exist per symbol. BUT `rl_gate_pass=false`, `rl_weight_effective=0.0` on ALL signals → RL has 0% influence on execution. |

---

## 8. TOP 10 BREAKPOINTS (prioritized by severity)

---

### 🚨 BREAKPOINT #1 — SL/TP SELL-SIDE MATH BUG
**Severity: CRITICAL**  
**Layer: L3 AI Engine**

**Evidence from `journalctl -u quantum-ai-engine`:**
```
WARNING [SL_DEBUG] BNBUSDT SELL: price=636.71, sl_pct=8.9893, tp_pct=17.9786
  → SL_CALC=(1+8.9893)=6360.287203   ← WRONG for SELL
  → TP_CALC=(1-17.9786)=-10810.444  ← NEGATIVE TP (impossible)
```

**Evidence from `quantum:stream:trade.intent` (XREVRANGE):**
```json
{
  "symbol": "BTCUSDT", "side": "SELL",
  "stop_loss": 96106059.76279199,     ← $96 MILLION stop loss
  "take_profit": -192011872.09058398  ← negative take profit
}
```

**Root cause:** For SELL orders, `stop_loss` should be calculated as `entry_price * (1 + sl_pct/100)` (price goes UP to stop you out), not `(1 - sl_pct)`. The code appears to be using the BUY formula on both sides.

**File to fix:**
```
/home/qt/quantum_trader/microservices/ai_engine/main.py
```
Search for: `stop_loss`, `take_profit`, `sl_pct`, `tp_pct` near SELL-side calculation.

---

### 🚨 BREAKPOINT #2 — EXECUTION CONSUMER GROUP STUCK (~15 DAYS)
**Severity: CRITICAL**  
**Layer: L6 Execution**

**Evidence from `redis-cli XINFO GROUPS quantum:stream:trade.intent`:**
```
Group:             quantum:group:execution:trade.intent
consumers:         1
pending:           0
last-delivered-id: 1771160145484-0   ← approx Feb 16, 2026
current stream head:                   1772501213711-0  ← Mar 3, 2026
Stream XLEN:       10,004 entries
lag:               NULL              ← checkpoint trimmed out of stream
```

**Evidence from `quantum:stream:trade.execution.res` (XREVRANGE):**
```
Last entry: 2026-02-09T23:20 — status: "rejected"
```
No executions since Feb 9. No intents consumed by execution group since ~Feb 16. 10,004 trade intents have piled up unprocessed.

**Likely causes:**
1. Redis restarted Mar 2 06:37 — blocking XREADGROUP call dropped, consumer didn't reconnect
2. Or: execution service consumer group offset was manually reset/lost

**File to inspect:**
```
/home/qt/quantum_trader/services/execution_service.py
```
Look for: `XREADGROUP quantum:group:execution:trade.intent`, consumer reconnection logic, error handling on Redis disconnect.

**Fix approach (READ-ONLY audit — do not apply yet):**
```bash
# Reset consumer group to current head (skip backlog):
redis-cli XGROUP SETID quantum:stream:trade.intent quantum:group:execution:trade.intent '$'
# Then restart execution service:
systemctl restart quantum-execution.service
```

---

### 🚨 BREAKPOINT #3 — EXITBRAIN V3.5 DISABLED + BROKEN DROP-IN
**Severity: CRITICAL**  
**Layer: L6 Execution**

**Evidence:**
```bash
systemctl status quantum-exitbrain-v35.service
→ inactive (dead), DISABLED

journalctl:
Mar 03 01:35:31 systemd[1]: /etc/systemd/system/quantum-exitbrain-v35.service.d/shadow.conf:3:
  Invalid syntax, ignoring: \EXIT_SHADOW_MODE=true
```

**Broken drop-in content** (`/etc/systemd/system/quantum-exitbrain-v35.service.d/shadow.conf`):
```ini
[Service]
[Service]                        ← duplicate section header
Environment=\EXIT_SHADOW_MODE=true\  ← invalid backslash syntax
```

**Consequence:** The primary exit decision engine (ExitBrain v3.5) is NOT running. Exit decisions for live positions are not being made. Only `exit_monitor_service_v2.py` (watchdog) is active — it monitors heartbeats and triggers panic close, but does NOT make intelligent TP/SL decisions.

**Files to fix:**
```
/etc/systemd/system/quantum-exitbrain-v35.service.d/shadow.conf
/etc/systemd/system/quantum-exitbrain-v35.service.d/control.conf
/etc/quantum/exitbrain-v35.env   (check contents)
```

---

### 🚨 BREAKPOINT #4 — POSITION STATE BRAIN CRASH LOOP (6568 RESTARTS)
**Severity: HIGH**  
**Layer: L6 Execution**

**Evidence from `journalctl -u quantum-position-state-brain`:**
```
Mar 03 01:33:10 python3[2751530]: 2026-03-03 01:33:10 [ERROR] Missing Binance credentials - P3.3 cannot function
Mar 03 01:33:10 systemd[1]: quantum-position-state-brain: Failed with result 'exit-code'
Mar 03 01:33:20 systemd[1]: Scheduled restart job, restart counter is at 6568.
```

**Unit file details:**
```ini
User=root        ← WRONG (should be qt)
ExecStart=/usr/bin/python3 ... ← system python, not venv
EnvironmentFile=/etc/quantum/position-state-brain.env  ← missing BINANCE_API_KEY?
```

**Problems:**
1. Running as `root` (security issue, inconsistent with other services)
2. Using `/usr/bin/python3` (system Python) instead of `/opt/quantum/venvs/ai-engine/bin/python3`
3. `BINANCE_API_KEY` missing from `/etc/quantum/position-state-brain.env`
4. CPU waste: ~6 restart attempts per minute continuously

**Files to fix:**
```
/etc/systemd/system/quantum-position-state-brain.service   (User, ExecStart)
/etc/quantum/position-state-brain.env                       (add Binance credentials)
```

---

### ⚠️ BREAKPOINT #5 — REDIS RESTARTED 19h AGO (VPS UPTIME MISMATCH)
**Severity: HIGH**  
**Layer: L0 Infrastructure**

**Evidence:**
```
VPS uptime:   42 days, 02:47 (since ~Jan 19 2026)
Redis uptime: 68,393 seconds = ~19 hours → restarted 2026-03-02 06:37 UTC
Exit-monitor started: 2026-03-02 06:37:02 UTC (same second)
→ Redis crash likely triggered a cascade restart
```

**Consequence:** Any services doing blocking XREAD/XREADGROUP lost their connection when Redis restarted. Services that auto-reconnect resumed correctly (intent_bridge — lag=0). Services that did not (execution consumer — lag=NULL) are now stuck.

**Files to inspect:**
```
/etc/redis/redis.conf                 (persistence: appendonly, maxmemory)
/var/log/redis/redis-server.log       (if exists — find crash reason)
journalctl -u redis                   (if redis is managed by systemd)
```

---

### ⚠️ BREAKPOINT #6 — EXPOSURE BALANCER + BSC DEAD SINCE FEB 21
**Severity: HIGH**  
**Layer: L4 Risk**

**Evidence:**
```
quantum-exposure_balancer.service:
  Active: inactive (dead) since Sat 2026-02-21 06:34:26 UTC
  Duration: 178ms → crash on startup (exit-code, status=1/FAILURE)

quantum-bsc.service:
  Active: inactive (dead) since Sat 2026-02-21 06:34:26 UTC
  Duration: 1h 22min → killed by SIGTERM

Both died at the EXACT same second: 06:34:26
```

**Consequence:**
- `exposure_balancer` enforces: max_margin_util=85%, max_symbol_exposure=15%, min_diversification=5 symbols — **these guardrails are NOT active**
- `bsc` (Baseline Safety Controller) — another safety layer — also dead

**Files to inspect:**
```
journalctl -u quantum-exposure_balancer --since "2026-02-21" --no-pager
journalctl -u quantum-bsc --since "2026-02-21" --no-pager
/home/qt/quantum_trader/microservices/exposure_balancer/
/etc/systemd/system/quantum-exposure_balancer.service
```

---

### ⚠️ BREAKPOINT #7 — NGINX CONFIG POLLUTION
**Severity: MEDIUM**  
**Layer: L0 NGINX**

**Evidence (nginx -t):**
```
[warn] conflicting server name "app.quantumfond.com" on 0.0.0.0:443, ignored (×5)
[warn] conflicting server name "app.quantumfond.com" on 0.0.0.0:80, ignored  (×5)
```

**Cause:** 6 files in `/etc/nginx/sites-enabled/` all declare `server_name app.quantumfond.com`. The `.backup`, `.bak`, `.backup2`, `.backup3`, `.backup_jan16` files are NOT symlinks but regular files loaded as config.

**Fix (READ-ONLY — do not apply yet):**
```bash
# Remove stale files from sites-enabled (move to archive):
cd /etc/nginx/sites-enabled/
mv app.quantumfond.com.backup   /etc/nginx/archive/
mv app.quantumfond.com.backup2  /etc/nginx/archive/
mv app.quantumfond.com.backup3  /etc/nginx/archive/
mv app.quantumfond.com.backup_jan16 /etc/nginx/archive/
mv app.quantumfond.com.bak      /etc/nginx/archive/
nginx -t && systemctl reload nginx
```

---

### ⚠️ BREAKPOINT #8 — RL LAYER FULLY GATED (rl_gate_pass=false)
**Severity: MEDIUM**  
**Layer: L8 RL**

**Evidence from `quantum:stream:trade.intent` (last 2 events):**
```json
"rl_influence_enabled": false,
"rl_gate_pass": false,
"rl_gate_reason": "rl_disabled",
"rl_weight_effective": 0.0,
"rl_effect": "none"
```

**Context:** services `quantum-rl-agent`, `quantum-rl-trainer`, `quantum-rl-feedback-v2`, `quantum-rl-calibrator` are all **active/running**. Keys `quantum:rl:policy:BNBUSDT`, `quantum:rl:state:*`, `quantum:rl:calibration:v1:*` all exist. The RL infrastructure is running but a kill-switch in ai-engine is disabling its influence.

**File to inspect:**
```
/etc/quantum/ai-engine.env         (look for RL_ENABLED=false or similar)
/home/qt/quantum_trader/microservices/ai_engine/main.py  (RL gate logic)
```

---

### ⚠️ BREAKPOINT #9 — MARKET REGIME ALWAYS "UNKNOWN"
**Severity: MEDIUM**  
**Layer: L3/L5**

**Evidence from `quantum:stream:trade.intent` (multiple events):**
```json
"regime": "unknown"
```

`quantum-meta-regime.service` is **active/running**, but the AI engine is not receiving or not reading its output. The regime should be one of: `sideways_wide`, `trending_bull`, `trending_bear`, `high_volatility`, etc.

**Impact:** Strategy selection defaults to fallback mode. In the log: `"meta_strategy": "default"` — no regime-adaptive strategy selection.

**Files to inspect:**
```
/home/qt/quantum_trader/microservices/meta_regime/       (what key does it write?)
/home/qt/quantum_trader/microservices/ai_engine/main.py  (what key does it read?)
redis-cli KEYS "quantum:regime:*"                        (does the key exist?)
redis-cli KEYS "quantum:market:regime"                   (alternative key name?)
```

---

### ⚠️ BREAKPOINT #10 — 20+ UNDOCUMENTED PROCESSES ON PUBLIC PORTS
**Severity: MEDIUM**  
**Layer: L0 Infrastructure**

**Evidence (ss -lntp):**
```
0.0.0.0:8002   python3 execution_service.py  (pid=1732234)  — public
0.0.0.0:8005   python3 unknown               (pid=1732086)  — unknown
0.0.0.0:8007   python3 exit-monitor          (pid=1732235)  — public
0.0.0.0:8020   python3 unknown               (pid=1732079)  — unknown
0.0.0.0:8042-8069  multiple python3 processes (12+ ports)   — undocumented
0.0.0.0:9092   python3 unknown               (pid=1732083)  — unknown
```

All bound to `0.0.0.0` (publicly reachable). None of these (except 8000 backend) are proxied through nginx with authentication.

System load average 3.28 correlates with the number of active python3 processes. This many open ports also suggests **process sprawl** — services that should share a port or use internal-only binding.

**Command to run to identify:**
```bash
ps aux | grep python3 | grep -v grep | awk '{print $2, $11, $12, $13}' | sort -k2
```

---

## 9. ACTIVE SIGNAL SAMPLE (evidence of live AI operation)

From `quantum:stream:trade.intent` entry `1772501213711-0` (2026-03-03T01:26:50 UTC):

```json
{
  "symbol": "APTUSDT",
  "side": "BUY",
  "position_size_usd": 100.0,
  "leverage": 10,
  "entry_price": 0.9495,
  "stop_loss": 0.9257625,       ← looks correct for BUY
  "take_profit": 0.9969750,     ← looks correct for BUY
  "confidence": 0.72,
  "model": "ensemble",
  "consensus_count": 1,
  "total_models": 4,
  "model_breakdown": {
    "xgb":      {"action": "HOLD", "confidence": 0.9478},
    "lgbm":     {"action": "BUY",  "confidence": 0.9000},
    "nhits":    {"action": "SELL", "confidence": 1.0000},
    "patchtst": {"action": "HOLD", "confidence": 0.4511},
    "tft":      {"action": "HOLD", "confidence": 0.6828},
    "fallback": {"action": "BUY",  "reason": "consensus_not_met_or_hold_signal", "triggered_by": "testnet_hash_pattern"}
  },
  "regime": "unknown",
  "rl_gate_pass": false,
  "rl_weight_effective": 0.0,
  "ai_leverage": 8.6,
  "ai_size_usd": 200.0
}
```

> Note: `consensus_count=1` with `total_models=4` — only 1 of 4 models agreed on BUY. The `fallback` triggered due to `consensus_not_met`. This is a low-conviction signal.

---

## 10. CRITICAL PATH FLOW STATUS

```
[Binance WS] → market.klines stream  → [AI Engine]  ✅
                                                 ↓
[AI Engine]  → quantum:stream:trade.intent        ✅ (10,004 signals queued)
                                                 ↓
[quantum:group:intent_bridge] → intent-bridge     ✅ lag=0
                                                 ↓
[quantum:group:execution:trade.intent]            🚨 STUCK since Feb 16
                                                 ↓
[Execution Service] → Binance REST API            ❌ last execution Feb 9

[ExitBrain v3.5]  → exit decisions               ❌ DISABLED
[Exposure Balancer] → margin guardrail            ❌ DEAD since Feb 21
[Position State Brain] → position state tracking 🔴 crash-looping
```

---

## 11. ENVIRONMENT FILES ON VPS (names only, no values)

Located at `/etc/quantum/`:
- `ai-engine.env` — ai-engine, execution service
- `autonomous-trader.env` — autonomous trader
- `exitbrain-v35.env` — ExitBrain v3.5 (includes Binance credentials)
- `exitbrain-control.env` — ExitBrain control layer drop-in
- `testnet.env` — testnet Binance credentials
- `intent-executor.env` — intent executor service
- `apply-layer.env` — apply layer + position-state-brain
- `position-state-brain.env` — P3.3 position state brain (INCOMPLETE — missing Binance creds)

---

## 12. RECOMMENDED FIXES (PRIORITY ORDER)

> **⚠️ All fixes below are READ-ONLY proposals from this audit. Apply carefully and test one at a time.**

| Priority | Fix | Command / File | Impact |
|----------|-----|---------------|--------|
| P1 | Fix execution consumer group stuck offset | `redis-cli XGROUP SETID quantum:stream:trade.intent quantum:group:execution:trade.intent '$'` then `systemctl restart quantum-execution.service` | Resumes trade execution |
| P1 | Enable ExitBrain v3.5 | Fix `/etc/systemd/system/quantum-exitbrain-v35.service.d/shadow.conf` + `systemctl enable --now quantum-exitbrain-v35.service` | Restores exit management |
| P1 | Fix SL/TP SELL-side math bug | `/home/qt/quantum_trader/microservices/ai_engine/main.py` — fix `(1 + sl_pct)` → `(1 - sl_pct)` for SELL | Prevents catastrophic order sizing |
| P2 | Fix position-state-brain | Add Binance creds to `/etc/quantum/position-state-brain.env`, fix `User=qt`, fix ExecStart to use venv python | Stops 6568-count crash loop |
| P2 | Restart exposure_balancer + bsc | Debug crash cause then `systemctl start quantum-exposure_balancer.service quantum-bsc.service` | Restores margin guardrails |
| P3 | Clean nginx sites-enabled | Remove `.backup*` and `.bak` files from `/etc/nginx/sites-enabled/` | Eliminates 10 config warnings |
| P3 | Investigate Redis restart cause | Check `/var/log/redis/`, `journalctl -u redis` | Prevent future cascades |
| P4 | Fix or enable RL gate | Check `RL_ENABLED` flag in `/etc/quantum/ai-engine.env` | Re-enables RL influence |
| P4 | Fix regime detection | Check key mismatch between meta-regime writer and ai-engine reader | Enables regime-adaptive strategies |
| P5 | Identify undocumented processes | `ps aux | grep python3` — identify ports 8005, 8020, 8042–8069, 9092 | Reduce attack surface + CPU load |

---

## APPENDIX — OBSERVABILITY STACK (confirmed running)

| Service | Port | Status |
|---------|------|--------|
| Prometheus | 9091 | ✅ active |
| Grafana | 3000 | ✅ active |
| Loki | 3100, 9095 | ✅ active |
| Promtail | 9080, 44015 | ✅ active |
| Node Exporter | 9100 | ✅ active |
| Redis Exporter | 9121 | ✅ active |

Accessible at: `https://app.quantumfond.com/grafana/`

---

*End of Forensic Audit Report — 2026-03-03*
