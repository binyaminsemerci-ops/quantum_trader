# Quantum Trader — Service Execution Pipeline Audit

**Date:** 2026-03-13  
**VPS:** 46.224.116.254  
**All 14 services: ACTIVE (running)**

---

## EXECUTIVE SUMMARY

| Category | Services | Count |
|----------|----------|-------|
| **TESTNET — places real orders on Binance TESTNET** | execution, intent-executor, apply-layer, exitbrain-v35, dag3-hw-stops, paper-trade-controller | 6 |
| **⚠️ PRODUCTION credentials loaded but limited scope** | emergency-exit-worker | 1 |
| **SHADOW/READ-ONLY — no orders placed** | exit-management-agent, exit-brain-shadow, exit-intent-gateway*, harvest-brain*, harvest-brain-2, risk-brake | 6 |
| **STREAM-ONLY — writes to Redis, no direct Binance** | autonomous-trader | 1 |

\* exit-intent-gateway and harvest-brain are "enabled/live" but only write to Redis streams, never directly to Binance.

---

## CRITICAL FINDING: PRODUCTION CREDENTIALS

The `.env` file at `/home/qt/quantum_trader/.env` contains:
```
BINANCE_API_KEY=e9ZqWhGh...      ← PRODUCTION key
BINANCE_API_SECRET=ZowBZEfL...   ← PRODUCTION secret
BINANCE_TESTNET=false             ← PRODUCTION flag
TESTNET=true                      ← contradicts above
```

**`quantum-emergency-exit-worker.service`** loads this `.env` via `EnvironmentFile=/home/qt/quantum_trader/.env`.  
It reads `BINANCE_TESTNET` (=false) → connects to **PRODUCTION** Binance.  
It reads `BINANCE_API_KEY` → uses the **PRODUCTION** API key.  
**This is the only service with production Binance credentials.**

---

## DETAILED SERVICE AUDIT

### ═══════════════════════════════════════
### ENTRY SIDE (Opening Positions)
### ═══════════════════════════════════════

#### 1. quantum-execution.service
| Field | Value |
|-------|-------|
| **ExecStart** | `/opt/quantum/venvs/ai-engine/bin/python3 services/execution_service.py` |
| **WorkingDirectory** | `/home/qt/quantum_trader` |
| **Binance Target** | **TESTNET** (`testnet=True`, URL=`testnet.binancefuture.com`) |
| **API Keys** | Testnet keys from `/etc/quantum/testnet.env` (w2W60kzu...) |
| **Places Orders?** | ✅ **YES** — `binance_client.futures_create_order()` at line 855 |
| **Reads Streams** | `quantum:stream:trade.intent`, `quantum:stream:apply.result` |
| **Writes Streams** | N/A (consumes intents, executes on Binance) |
| **Mode** | **TESTNET LIVE** — executes real orders on testnet |

#### 2. quantum-intent-executor.service
| Field | Value |
|-------|-------|
| **ExecStart** | `/usr/bin/python3 -m microservices.intent_executor.main` |
| **WorkingDirectory** | `/home/qt/quantum_trader` |
| **Binance Target** | **TESTNET** (`BINANCE_BASE_URL=https://testnet.binancefuture.com`) |
| **API Keys** | Testnet keys from `/etc/quantum/intent-executor.env` |
| **Places Orders?** | ✅ **YES** — `/fapi/v1/order` POST at lines 574, 627 |
| **Reads Streams** | `quantum:stream:apply.plan`, `quantum:stream:apply.plan.manual`, `quantum:stream:harvest.intent` |
| **Writes Streams** | `quantum:stream:apply.result`, `quantum:stream:trade.closed` |
| **Mode** | **TESTNET LIVE** — executes ENTRY + CLOSE orders on testnet |

#### 3. quantum-autonomous-trader.service
| Field | Value |
|-------|-------|
| **ExecStart** | `/opt/quantum/venvs/ai-client-base/bin/python -u microservices/autonomous_trader/autonomous_trader.py` |
| **WorkingDirectory** | `/home/qt/quantum_trader` |
| **Binance Target** | **NONE** — does NOT call Binance directly |
| **Places Orders?** | ❌ **NO** — only writes intents to Redis streams |
| **Writes Streams** | `quantum:stream:trade.intent` (entry), `quantum:stream:harvest.intent` (exit) |
| **Reads** | Redis position keys, AI Engine API (localhost:8001) |
| **Mode** | **TESTNET** (`BINANCE_USE_TESTNET=true` in env) — intent producer only |
| **Note** | `_monitor_positions()` is **delegated** to exit_management_agent when `quantum:exit_agent:active_flag` exists (PATCH-2) |

#### 4. quantum-paper-trade-controller.service
| Field | Value |
|-------|-------|
| **ExecStart** | `/opt/quantum/venvs/ai-client-base/bin/python paper_trade_controller.py` |
| **WorkingDirectory** | `/opt/quantum/microservices/paper_trade_controller` |
| **Binance Target** | **TESTNET** (hardcoded `https://testnet.binancefuture.com`) |
| **API Keys** | Testnet keys (`BINANCE_TESTNET_API_KEY`) — **currently empty in service file** |
| **Places Orders?** | ✅ **YES** (testnet) — but falls back to simulated fills when no creds |
| **Reads Streams** | `quantum:stream:harvest.v2.shadow` |
| **Writes Streams** | `quantum:stream:trade.closed` (tagged `paper_trade=true`) |
| **Mode** | **TESTNET** — Phase 2 paper trading on testnet |

#### 5. quantum-apply-layer.service
| Field | Value |
|-------|-------|
| **ExecStart** | `/usr/bin/python3 -u microservices/apply_layer/main.py` |
| **WorkingDirectory** | `/home/qt/quantum_trader` |
| **Binance Target** | **TESTNET** (hardcoded `https://testnet.binancefuture.com`) |
| **API Keys** | Testnet keys via `/etc/quantum/testnet.env` |
| **Places Orders?** | ✅ **YES** — `/fapi/v1/order` POST at lines 392, 716 |
| **Reads Streams** | `quantum:stream:apply.plan`, `quantum:stream:reconcile.close` |
| **Writes Streams** | `quantum:stream:apply.result`, `quantum:stream:trade.closed`, `quantum:stream:apply.heat.observed` |
| **Mode** | **TESTNET** (`APPLY_MODE=testnet`) — no "live" mode exists in code |

### ═══════════════════════════════════════
### EXIT SIDE (Closing Positions)
### ═══════════════════════════════════════

#### 6. quantum-exit-management-agent.service
| Field | Value |
|-------|-------|
| **ExecStart** | `/home/qt/quantum_trader_venv/bin/python -m microservices.exit_management_agent.main` |
| **WorkingDirectory** | `/home/qt/quantum_trader` |
| **Binance Target** | **NONE** — does NOT call Binance directly |
| **Places Orders?** | ❌ **NO** — writes exit intents to streams only |
| **Reads** | Redis position keys (positionRisk via Binance read-only) |
| **Writes Streams** | `quantum:stream:exit.audit`, `quantum:stream:exit.metrics`, `quantum:stream:exit.intent`, `quantum:stream:exit.outcomes`, `quantum:stream:exit.replay` |
| **FORBIDDEN Streams** | `apply.plan`, `trade.intent`, `harvest.intent`, `harvest.suggestions` (write guards enforced) |
| **Mode** | **LIVE WRITES to exit.intent** (`EXIT_AGENT_LIVE_WRITES_ENABLED=true`) but DRY_RUN=true (no direct execution). SCORING_MODE=ensemble. AI Judge in shadow. |
| **Note** | Sets `quantum:exit_agent:active_flag` (PATCH-6: ownership transfer from AT) |

#### 7. quantum-exitbrain-v35.service
| Field | Value |
|-------|-------|
| **ExecStart** | `/opt/quantum/venvs/ai-engine/bin/python3 microservices/position_monitor/main_exitbrain.py` |
| **WorkingDirectory** | `/home/qt/quantum_trader` |
| **Binance Target** | **TESTNET** (`BINANCE_TESTNET=true` → `testnet.binancefuture.com`) |
| **API Keys** | Testnet keys from `/etc/quantum/exitbrain-v35.env` |
| **Places Orders?** | ✅ **YES** — `futures_create_order()` for TP/SL/emergency at lines 258, 882, 946, 1209, 1246, 1280, 1318, 1358, 1862 |
| **Reads** | Binance positionRisk, Redis position keys |
| **Mode** | **TESTNET LIVE** (`EXIT_EXECUTOR_MODE=LIVE`, `EXIT_BRAIN_V3_LIVE_ROLLOUT=ENABLED`) |
| **Note** | Most aggressive exit service — places TP, SL, trailing stops, emergency closes directly on testnet |

#### 8. quantum-exit-brain-shadow.service
| Field | Value |
|-------|-------|
| **ExecStart** | `/home/qt/quantum_trader_venv/bin/python3 -m microservices.exit_brain_v1.run_exit_brain` |
| **WorkingDirectory** | `/home/qt/quantum_trader` |
| **Binance Target** | **NONE** — pure shadow/read-only |
| **Places Orders?** | ❌ **NO** — hardcoded `shadow_only=True` on all data contracts |
| **Writes Streams** | `quantum:stream:exit.state.shadow`, `exit.geometry.shadow`, `exit.regime.shadow`, `exit.ensemble.*.shadow`, `exit.belief.shadow`, `exit.hazard.shadow`, `exit.utility.shadow`, `exit.policy.shadow`, `exit.intent.candidate.shadow` (~17 shadow streams) |
| **FORBIDDEN Streams** | `trade.intent`, `apply.plan`, `apply.plan.manual`, `apply.result`, `exit.intent`, `harvest.intent` |
| **Mode** | **PURE SHADOW** — analysis only, writes to `.shadow` suffixed streams |

#### 9. quantum-exit-intent-gateway.service
| Field | Value |
|-------|-------|
| **ExecStart** | `/home/qt/quantum_trader_venv/bin/python3 -m microservices.exit_intent_gateway.main` |
| **WorkingDirectory** | `/home/qt/quantum_trader` |
| **Binance Target** | **NONE** — stream router only |
| **Places Orders?** | ❌ **NO** — forwards intents between streams only, never touches Binance |
| **Reads Streams** | `quantum:stream:exit.intent` (consumer group) |
| **Writes Streams** | `quantum:stream:trade.intent` (approved), `quantum:stream:exit.intent.rejected` (denied) |
| **Mode** | **ENABLED** (`EXIT_GATEWAY_ENABLED=true`) with `TESTNET_MODE=true` safety guard hardcoded AFTER env file |
| **Note** | This is the bridge from exit_management_agent → execution pipeline. Validates, deduplicates, rate-limits, cooldown-checks exit intents. |

#### 10. quantum-harvest-brain.service
| Field | Value |
|-------|-------|
| **ExecStart** | `/opt/quantum/venvs/ai-client-base/bin/python -u harvest_brain.py` |
| **WorkingDirectory** | `/opt/quantum/microservices/harvest_brain` |
| **Binance Target** | **TESTNET** (reads from `testnet.binancefuture.com` for positionRisk/klines) |
| **Places Orders?** | ❌ **NO** — only writes to Redis streams |
| **Reads** | Binance testnet positionRisk (read-only), `quantum:stream:apply.result` |
| **Writes Streams** | `quantum:stream:harvest.suggestions` (shadow), attempts `quantum:stream:apply.plan` (live) |
| **Mode** | **LIVE** (`HARVEST_MODE=live`) BUT `HARVEST_LIVE_WRITES_ENABLED` is **NOT SET** → apply.plan writes are **suppressed** → routes to shadow stream |
| **Effective Mode** | **SHADOW** despite HARVEST_MODE=live |

#### 11. quantum-harvest-brain-2.service
| Field | Value |
|-------|-------|
| **ExecStart** | Same as harvest-brain (same .py file) |
| **Binance Target** | **TESTNET** |
| **Places Orders?** | ❌ **NO** |
| **Mode** | **EXPLICIT SHADOW** (`EXIT_SHADOW_MODE=true`, `HARVEST_SHADOW_MODE=true`) |
| **Restart** | `Restart=no` — single consumer test instance |

#### 12. quantum-risk-brake.service
| Field | Value |
|-------|-------|
| **ExecStart** | `/opt/quantum/venvs/harvest-v2/bin/python risk_brake_v1_patch.py` |
| **WorkingDirectory** | `/opt/quantum/microservices` |
| **Source File** | **DELETED from disk** — running from memory since 2026-02-22 |
| **Places Orders?** | ❓ **UNKNOWN** (source deleted), but log output shows `emitted=0` consistently |
| **Mode** | Appears **READ-ONLY monitoring** — scans 33 positions per cycle, emits nothing |
| **Note** | ⚠️ Running for 19 days from deleted binary. Should be investigated/redeployed. |

#### 13. quantum-emergency-exit-worker.service
| Field | Value |
|-------|-------|
| **ExecStart** | `/opt/quantum/venvs/ai-client-base/bin/python emergency_exit_worker.py` |
| **WorkingDirectory** | `/home/qt/quantum_trader` |
| **Binance Target** | ⚠️ **PRODUCTION** (`BINANCE_TESTNET=false` in `.env`) |
| **API Keys** | ⚠️ **PRODUCTION** keys (`e9ZqWhGh...`) |
| **Places Orders?** | ✅ **YES** — `futures_create_order()` at line 376 |
| **Reads Streams** | `system:panic_close` (consumer group) |
| **Writes Streams** | `system:panic_close:completed` |
| **Mode** | ⚠️ **PRODUCTION LIVE** — responds to panic_close events with real market close orders |
| **Restart** | `Restart=no` — manual inspection required on failure |
| **CRITICAL** | Only fires on `system:panic_close` stream events. Dormant unless panic is triggered. |

#### 14. quantum-dag3-hw-stops.service
| Field | Value |
|-------|-------|
| **ExecStart** | `/opt/quantum/venvs/ai-client-base/bin/python -u dag3_hw_stops.py` |
| **WorkingDirectory** | `/opt/quantum/microservices/dag3_hw_stops` |
| **Binance Target** | **TESTNET** (hardcoded `https://testnet.binancefuture.com`) |
| **API Keys** | Testnet keys from `/etc/quantum/testnet.env` |
| **Places Orders?** | ✅ **YES** — STOP_MARKET and TAKE_PROFIT_MARKET orders at lines 148, 162 |
| **Reads** | Redis position keys, Binance openOrders (testnet) |
| **Mode** | **TESTNET LIVE** — places/maintains hardware TP/SL orders |
| **Killswitch** | `redis-cli SET quantum:dag3:hw_stops:disabled 1` |

---

## ORDER FLOW MAP

```
                    ┌──────────────────────┐
                    │  autonomous_trader   │ (RL signals)
                    │  NO direct Binance   │
                    └──────┬───────┬───────┘
                           │       │
              ENTRY intents│       │EXIT intents
                           ▼       ▼
              trade.intent    harvest.intent
                   │               │
                   ▼               ▼
         ┌─────────────┐   ┌──────────────────┐
         │ execution   │   │ intent_executor  │
         │  service    │   │ (ENTRY+CLOSE)    │
         │ TESTNET ✅  │   │ TESTNET ✅       │
         └─────────────┘   └──────────────────┘
                                   ▲
                                   │ apply.plan
                                   │
              ┌────────────────────┤
              │                    │
     ┌────────────────┐    ┌──────────────┐
     │  apply_layer   │    │ harvest_brain│
     │ TESTNET ✅     │    │ SHADOW(eff.) │
     │ reads apply.plan│    │ writes blocked│
     └────────────────┘    └──────────────┘

  EXIT PATH (EMA pipeline):
  exit_mgmt_agent → exit.intent → exit_intent_gateway → trade.intent → execution_service
       (DRY_RUN)     (validates)     (forwards if OK)                    (TESTNET ✅)

  EXIT PATH (direct):
  exitbrain_v35 → futures_create_order() → TESTNET ✅ (TP/SL/emergency)
  dag3_hw_stops → STOP_MARKET/TP orders → TESTNET ✅

  EMERGENCY PATH:
  system:panic_close → emergency_exit_worker → futures_create_order() → ⚠️ PRODUCTION
```

---

## CONFLICT ANALYSIS

### ⚠️ CONFLICT 1: Dual Exit Executors on Testnet
- **exitbrain_v35** places TP/SL/emergency orders directly via `futures_create_order()`
- **execution_service** also executes exit orders from `trade.intent` stream
- **intent_executor** also executes closes from `apply.plan` and `harvest.intent`
- **dag3_hw_stops** independently places STOP_MARKET orders
- **Result:** 4 services may compete to close the same testnet position, potentially causing duplicate close attempts and "position not found" errors.

### ⚠️ CONFLICT 2: Dual Entry Paths to trade.intent
- **autonomous_trader** writes ENTRY intents to `quantum:stream:trade.intent`
- **exit_intent_gateway** also forwards EXIT intents to the same `quantum:stream:trade.intent`
- **execution_service** consumes from `trade.intent` — it sees both ENTRY and EXIT mixed in one stream.

### ⚠️ CONFLICT 3: Production Emergency Path
- **emergency_exit_worker** uses **PRODUCTION** credentials (`BINANCE_TESTNET=false` in `.env`)
- All other order-placing services use **TESTNET** credentials
- If `system:panic_close` is triggered, it will attempt to close positions on **PRODUCTION** Binance
- **But** production positions don't exist (all trading is testnet) → orders would fail

### ⚠️ CONFLICT 4: harvest-brain Mode Confusion
- `HARVEST_MODE=live` is set (suggests live)
- But `HARVEST_LIVE_WRITES_ENABLED` is not set → writes to `apply.plan` are suppressed
- Effective mode is shadow despite the `live` label — confusing for operators

### ⚠️ CONCERN 5: Risk Brake Running from Deleted File
- `risk_brake_v1_patch.py` was deleted from disk but process runs since Feb 22
- Cannot inspect or update the running code
- Should be redeployed or stopped

---

## SERVICES THAT ACTUALLY PLACE REAL ORDERS

### On TESTNET:
1. **execution_service** — ENTRY orders (from trade.intent)
2. **intent_executor** — ENTRY + CLOSE orders (from apply.plan, harvest.intent)  
3. **apply_layer** — ENTRY + CLOSE orders (from apply.plan, reconcile.close)
4. **exitbrain_v35** — EXIT orders (TP/SL/trailing/emergency directly)
5. **dag3_hw_stops** — EXIT orders (hardware TP/SL STOP_MARKET)
6. **paper_trade_controller** — ENTRY orders (from harvest.v2.shadow, simulated fills if no creds)

### On PRODUCTION:
7. **emergency_exit_worker** — EXIT panic-close orders (**only on system:panic_close trigger**)

### No Orders (Shadow/Stream-only):
8. autonomous_trader — writes intents, no Binance
9. exit_management_agent — writes exit.intent, no Binance
10. exit_brain_shadow — pure shadow analysis
11. exit_intent_gateway — stream router, no Binance
12. harvest_brain — effectively shadow (writes suppressed)
13. harvest_brain_2 — explicit shadow mode
14. risk_brake — monitoring only (emitted=0, deleted binary)
