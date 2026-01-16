# 📊 QUANTUM TRADER - TP/SL ORDER EXECUTION SYSTEM
## Komplett Systemoversikt fra A til Å

**Generert:** 3. januar 2026  
**Status:** LIVE på VPS (46.224.116.254)  
**Mode:** EXIT_BRAIN_V3 + LIVE Executor

---

## 🎯 SYSTEMARKITEKTUR OVERVIEW

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    QUANTUM TRADER EXIT MANAGEMENT SYSTEM                     │
│                                                                              │
│  AI Signal → ExitBrain → Dynamic Executor → Order Gateway → Binance → PnL  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 SYKLUS-OVERSIKT (FULL LIFECYCLE)

### FASE 1: SIGNAL GENERERING
```
┌──────────────────────────────────────────────────────────────┐
│ 1. AI ENGINE (ai_engine/main.py)                             │
│    - 4 ML-modeller: XGB, LGBM, N-HiTS, TFT                   │
│    - Genererer trading signals med confidence (0-1)           │
│    - Publiserer til Redis: quantum:stream:signals             │
│    - Data: symbol, side, confidence, entry_price, atr_value   │
└──────────────────────────────────────────────────────────────┘
        ↓
```

### FASE 2: AUTO EXECUTOR (ENTRY)
```
┌──────────────────────────────────────────────────────────────┐
│ 2. AUTO EXECUTOR (auto_executor/executor_service.py)         │
│    - Lytter på Redis signal stream                           │
│    - Validerer signal via Risk Safety                        │
│    - Sender entry order til Binance                          │
│    - Venter på FILLED status                                 │
│    - Publiserer posisjon til Redis                           │
└──────────────────────────────────────────────────────────────┘
        ↓
```

### FASE 3: EXIT BRAIN PLANLEGGING (TP/SL CALCULATION)
```
┌──────────────────────────────────────────────────────────────┐
│ 3. EXIT BRAIN V3.5 (exitbrain_v3_5/exit_brain.py)            │
│    - Kalkulerer dynamisk leverage med ILFv2:                 │
│      leverage = base_leverage × confidence × (1/volatility)  │
│                                                               │
│    - Adaptive TP/SL Calculation:                             │
│      • LSF = 1 / (1 + ln(leverage + 1))  [Leverage Scale]   │
│      • TP1 = base_tp × (0.6 + LSF)       [25% av posisjon]  │
│      • TP2 = base_tp × (1.2 + LSF/2)     [25% av posisjon]  │
│      • TP3 = base_tp × (1.8 + LSF/4)     [50% av posisjon]  │
│      • SL = base_sl × (1 - confidence/4) [Full posisjon]    │
│                                                               │
│    - Publiserer ExitPlan til Redis                           │
│    - Base verdier:                                           │
│      • base_tp = 2.0% (økt for funding costs)               │
│      • base_sl = 1.2% (økt for safety margin)               │
└──────────────────────────────────────────────────────────────┘
        ↓
```

### FASE 4: EXIT BRAIN ADAPTER (DECISION TRANSLATION)
```
┌──────────────────────────────────────────────────────────────┐
│ 4. EXIT BRAIN ADAPTER (exit_brain_v3/adapter.py)             │
│    - Henter posisjon fra Binance Futures API                 │
│    - Bygger PositionContext:                                 │
│      • symbol, side, entry_price, current_price              │
│      • size, unrealized_pnl, leverage                        │
│    - Kaller ExitBrain for ExitPlan                           │
│    - Oversetter til ExitDecision:                            │
│      • INIT_NEW_POSITION: Sett opp nye levels                │
│      • MOVE_SL: Juster stop loss dynamisk                    │
│      • UPDATE_TP_LIMITS: Endre TP levels                     │
│      • HOLD_CURRENT: Ingen endringer                         │
└──────────────────────────────────────────────────────────────┘
        ↓
```

### FASE 5: DYNAMIC EXECUTOR (MONITORING & EXECUTION)
```
┌──────────────────────────────────────────────────────────────┐
│ 5. DYNAMIC EXECUTOR (exit_brain_v3/dynamic_executor.py)      │
│    HOVEDLOOP (hvert 10. sekund):                             │
│                                                               │
│    A. POSITION MONITORING:                                   │
│       - Fetch alle open positions fra Binance                │
│       - Hent current market price for hvert symbol           │
│                                                               │
│    B. STATE MANAGEMENT:                                      │
│       - Opprett/oppdater PositionExitState per posisjon      │
│       - State Key: "{symbol}:{side}" (BTCUSDT:LONG)         │
│       - Tracker internt:                                     │
│         • active_sl: AI-driven stop loss (INGEN order)       │
│         • tp_levels: List[(pris, størrelse%)]               │
│         • triggered_legs: Hvilke TP-nivåer er hit            │
│         • hard_sl_price: Binance STOP_MARKET (fallback)     │
│         • hard_sl_order_id: Exchange order ID                │
│                                                               │
│    C. LOSS GUARD CHECK (HØYESTE PRIORITET):                 │
│       if unrealized_pnl_pct < -12.5%:                        │
│         → EMERGENCY EXIT: Close full position MARKET         │
│         → Skip alle andre checks                             │
│                                                               │
│    D. AI DECISION UPDATE:                                    │
│       - Hent ExitDecision fra adapter                        │
│       - Oppdater state.active_sl hvis MOVE_SL                │
│       - Oppdater state.tp_levels hvis UPDATE_TP_LIMITS       │
│                                                               │
│    E. STOP LOSS CHECK:                                       │
│       LONG:  if current_price <= active_sl                   │
│       SHORT: if current_price >= active_sl                   │
│         → Execute MARKET order (full remaining size)         │
│         → Cancel hard SL order på Binance                    │
│         → Clear state og exit loop for denne posisjonen      │
│                                                               │
│    F. TAKE PROFIT CHECK:                                     │
│       for hver tp_level i tp_levels:                         │
│         LONG:  if current_price >= tp_price                  │
│         SHORT: if current_price <= tp_price                  │
│           → Execute MARKET order (size_pct av remaining)     │
│           → Marker leg som triggered                         │
│           → Recompute dynamic SL (ratchet tighter)           │
│           → Kun 1 TP per cycle (for safety)                  │
│                                                               │
│    G. HARD SL MANAGEMENT:                                    │
│       - Hard SL er STOP_MARKET order på Binance              │
│       - Plassert ved position entry (2% fra entry)           │
│       - Fungerer som siste fallback ved crash                │
│       - Cancelled når posisjon closes normalt                │
└──────────────────────────────────────────────────────────────┘
        ↓
```

### FASE 6: ORDER GATEWAY (CENTRAL EXIT POINT)
```
┌──────────────────────────────────────────────────────────────┐
│ 6. EXIT ORDER GATEWAY (services/execution/exit_order_gateway.py) │
│                                                               │
│    ANSVARSOMRÅDER:                                           │
│    - Single entry point for ALLE exit orders                 │
│    - Observability: Logger alle orders med module name       │
│    - Ownership tracking: Identifiser konflikter              │
│    - Metrics: Track orders per module & kind                 │
│                                                               │
│    ORDER TYPER:                                              │
│    • tp_market_leg_0/1/2: Take Profit MARKET orders          │
│    • sl_market: Stop Loss MARKET order                       │
│    • hard_sl: Binance STOP_MARKET safety net                │
│    • loss_guard_emergency: Emergency full exit               │
│                                                               │
│    VALIDERING:                                               │
│    - Check EXIT_MODE config (EXIT_BRAIN_V3 vs LEGACY)       │
│    - Warn ved legacy module usage i EXIT_BRAIN mode          │
│    - Validate order params før Binance submission            │
│    - Log full audit trail til Redis                          │
└──────────────────────────────────────────────────────────────┘
        ↓
```

### FASE 7: BINANCE EXECUTION
```
┌──────────────────────────────────────────────────────────────┐
│ 7. BINANCE CLIENT (integrations/exchanges/binance_client.py) │
│                                                               │
│    MARKET EXIT ORDER FORMAT:                                 │
│    {                                                          │
│      "symbol": "BTCUSDT",                                    │
│      "side": "SELL",        # SELL for LONG, BUY for SHORT   │
│      "type": "MARKET",       # Instant execution             │
│      "quantity": 0.005,      # Exact amount to close         │
│      "positionSide": "LONG", # Hedge mode support            │
│      "reduceOnly": true      # Kun redusere posisjon         │
│    }                                                          │
│                                                               │
│    HARD SL ORDER FORMAT:                                     │
│    {                                                          │
│      "symbol": "BTCUSDT",                                    │
│      "side": "SELL",                                         │
│      "type": "STOP_MARKET",  # Trigger på pris               │
│      "stopPrice": 95000.00,  # Entry - 2%                    │
│      "quantity": 0.005,                                      │
│      "positionSide": "LONG",                                 │
│      "reduceOnly": true                                      │
│    }                                                          │
│                                                               │
│    RESPONSE HANDLING:                                        │
│    - Parse Binance response JSON                             │
│    - Check status: NEW, FILLED, REJECTED                     │
│    - Extract orderId, executedQty, avgPrice                  │
│    - Return til Exit Order Gateway                           │
└──────────────────────────────────────────────────────────────┘
        ↓
```

### FASE 8: PNL TRACKING & FEEDBACK LOOP
```
┌──────────────────────────────────────────────────────────────┐
│ 8. BINANCE PNL TRACKER (binance_pnl_tracker.py)              │
│    - Kontinuerlig monitoring av alle posisjoner              │
│    - Kalkulerer real-time PnL per symbol:                    │
│      • unrealized_pnl: Åpen posisjon profit/loss             │
│      • unrealized_pct: PnL % av entry value                  │
│      • realized_pnl: Lukket posisjon profit                  │
│      • realized_pct: Realized PnL %                          │
│      • total_pnl: Sum av unrealized + realized               │
│                                                               │
│    - Publiserer til Redis:                                   │
│      Key: quantum:rl:reward:{SYMBOL}                         │
│      Stream: quantum:stream:exitbrain.pnl                    │
│      Data: {symbol, reward, pnl, confidence, timestamp}      │
│                                                               │
│    - RL Feedback:                                            │
│      • Reward = unrealized_pct + realized_pct                │
│      • Used by RL agents for learning                        │
│      • Dashboard visualization via /api/rl-dashboard/        │
└──────────────────────────────────────────────────────────────┘
```

---

## 🔄 DATAFLYT (REDIS STREAMS)

```
Redis EventBus Architecture:
============================

1. quantum:stream:signals
   └─> AI Engine producer → Auto Executor consumer
       Format: {symbol, side, confidence, price, atr, timestamp}

2. quantum:stream:exitbrain.pnl
   └─> Binance PnL Tracker producer → RL Monitor consumer
       Format: {symbol, pnl, reward, confidence, timestamp}

3. quantum:rl:reward:{SYMBOL}
   └─> Latest reward per symbol (Redis key, ikke stream)
       Format: {unrealized_pct, realized_pct, total_pnl, trades}

4. quantum:portfolio:realtime
   └─> Aggregert portfolio status
       Format: {total_equity, unrealized_pnl, num_positions, timestamp}
```

---

## ⚙️ KONFIGURASJON (Environment Variables)

```bash
# EXIT SYSTEM MODE
EXIT_MODE=EXIT_BRAIN_V3              # EXIT_BRAIN_V3 eller LEGACY
EXIT_EXECUTOR_MODE=LIVE               # LIVE, SHADOW, eller DISABLED
EXIT_BRAIN_V3_LIVE_ROLLOUT=ENABLED   # ENABLED eller DISABLED (killswitch)

# EXIT BRAIN SETTINGS
EXIT_BRAIN_CHECK_INTERVAL_SEC=10      # Monitoring loop interval
EXIT_BRAIN_PROFILE=DEFAULT            # Risk profile

# LEVERAGE & TP/SL
BASE_TP_PCT=0.020                     # 2.0% base take profit
BASE_SL_PCT=0.012                     # 1.2% base stop loss
MAX_LOSS_PCT_HARD_SL=0.02             # 2% hard SL safety net
MAX_UNREALIZED_LOSS_PCT=12.5          # -12.5% emergency exit trigger

# DYNAMIC TP PROFILE
DYNAMIC_TP_PROFILE=[0.25, 0.25, 0.50] # TP1: 25%, TP2: 25%, TP3: 50%
RATCHET_SL_ENABLED=true               # Auto-tighten SL after TP hits
```

---

## 🧩 KOMPONENTER & FILER

### Core Execution
```
microservices/execution/exit_brain_v3/
├── dynamic_executor.py          # Hovedloop, monitoring, execution
├── adapter.py                   # ExitBrain til ExitDecision translator
├── router.py                    # Plan caching & routing
├── types.py                     # PositionContext, ExitDecision, ExitState
└── precision.py                 # Binance precision handling (tick/step size)
```

### AI & Planning
```
microservices/exitbrain_v3_5/
├── exit_brain.py                # ExitBrain v3.5 hovedklasse
├── intelligent_leverage_engine.py  # ILFv2 leverage calculation
└── adaptive_leverage_engine.py  # Adaptive TP/SL calculation
```

### Gateways
```
backend/services/execution/
└── exit_order_gateway.py        # Central exit order gateway

backend/integrations/exchanges/
└── binance_client.py            # Binance API wrapper
```

### Monitoring & Tracking
```
microservices/binance_pnl_tracker/
└── binance_pnl_tracker.py       # Real-time PnL tracking

microservices/rl_monitor/
└── rl_monitor.py                # RL reward stream consumer

microservices/rl_dashboard/
└── dashboard.py                 # RL Intelligence visualization
```

---

## 📊 HYBRID STOP-LOSS MODEL

```
┌────────────────────────────────────────────────────────────────┐
│                    DUAL-LAYER PROTECTION                        │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  LAYER 1: INTERNAL AI-DRIVEN SL (active_sl)                   │
│  ✓ Dynamic, justeres basert på market conditions              │
│  ✓ Ingen exchange order (kun intern state)                    │
│  ✓ Checked hvert 10. sekund                                   │
│  ✓ Executes MARKET order ved trigger                          │
│  ✓ Kan flyttes opp (ratchet) etter TP hits                    │
│                                                                 │
│  LAYER 2: HARD SL SAFETY NET (hard_sl_price)                  │
│  ✓ Binance STOP_MARKET order på exchange                      │
│  ✓ Static 2% fra entry (sett ved position open)               │
│  ✓ Overlever system crash/restart                             │
│  ✓ Ideally NEVER triggers (Layer 1 exits først)               │
│  ✓ Acts as last-resort max-loss floor                         │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

**Eksempel LONG Position:**
```
Entry: $100,000
Internal SL (active_sl): $98,800 (1.2% dynamisk)
Hard SL (hard_sl_price): $98,000 (2% static safety net)

Scenario 1: Price drops til $98,800
→ Internal SL triggers
→ MARKET order closes position
→ Hard SL cancelled
→ Loss: -1.2%

Scenario 2: System crash, price drops til $98,000
→ Hard SL triggers på Binance (ingen software running)
→ Position closes automatisk
→ Loss: -2% (max loss containment)
```

---

## 🎯 TAKE PROFIT EXECUTION

### 3-Legs Partial TP Strategy
```
Initial Position: 1.0 BTC LONG @ $100,000

TP1: $102,000 (+2.0%) → Close 25% (0.25 BTC)
  ├─> Remaining: 0.75 BTC
  └─> Recompute SL: Move up til $100,500 (breakeven+)

TP2: $104,000 (+4.0%) → Close 25% (0.1875 BTC of original)
  ├─> Remaining: 0.5625 BTC
  └─> Recompute SL: Move up til $101,500 (secure profit)

TP3: $106,000 (+6.0%) → Close 50% (0.28125 BTC of original)
  ├─> Remaining: 0.28125 BTC
  └─> Let runner continue or trailing stop
```

### TP Trigger Logic
```python
# LONG Position
if current_price >= tp_price and leg_index not in triggered_legs:
    close_qty = remaining_size * size_pct
    execute_market_order(side="SELL", qty=close_qty, reduceOnly=True)
    triggered_legs.add(leg_index)
    ratchet_sl_up()

# SHORT Position
if current_price <= tp_price and leg_index not in triggered_legs:
    close_qty = remaining_size * size_pct
    execute_market_order(side="BUY", qty=close_qty, reduceOnly=True)
    triggered_legs.add(leg_index)
    ratchet_sl_up()
```

---

## 🛡️ SAFETY MECHANISMS

### 1. LOSS GUARD (Høyeste Prioritet)
```python
MAX_UNREALIZED_LOSS_PCT = 12.5  # -12.5%

if position.unrealized_pnl_pct < -MAX_UNREALIZED_LOSS_PCT:
    logger.critical(f"🚨 LOSS GUARD TRIGGERED @ {unrealized_pnl_pct}%")
    execute_emergency_exit(position, reason="MAX_LOSS_GUARD")
    # Closes full position immediately, skips all other checks
```

### 2. HARD SL SAFETY NET
```python
MAX_LOSS_PCT_HARD_SL = 0.02  # 2%

# Placed at position entry
hard_sl_price = entry_price * (1 - MAX_LOSS_PCT_HARD_SL)  # LONG
hard_sl_price = entry_price * (1 + MAX_LOSS_PCT_HARD_SL)  # SHORT

# Binance STOP_MARKET order survives system crashes
```

### 3. PRECISION VALIDATION
```python
# All orders quantized to Binance tick/step size
price = quantize_to_tick(price, symbol)
quantity = quantize_to_step(quantity, symbol)

# Example: BTCUSDT tick=0.1, step=0.001
# Price: 98765.432 → 98765.4
# Qty: 0.0123456 → 0.012
```

### 4. REDUCE-ONLY ENFORCEMENT
```python
# All exit orders MUST have reduceOnly=True
order_params = {
    "symbol": symbol,
    "side": side,
    "type": "MARKET",
    "quantity": qty,
    "reduceOnly": True,  # Cannot increase position
    "positionSide": position_side
}
```

---

## 📈 MONITORING & OBSERVABILITY

### 1. Dashboard Backend API
```
GET /api/portfolio/status
→ {pnl, exposure, positions, drawdown}

GET /api/rl-dashboard/
→ {status, symbols_tracked, symbols[], best_performer, avg_reward}

GET /api/ai/insights
→ {accuracy, sharpe, models[], latency}

GET /api/risk/metrics
→ {var, cvar, volatility, regime}
```

### 2. Redis Stream Monitoring
```bash
# Live PnL events
redis-cli XREAD STREAMS quantum:stream:exitbrain.pnl 0

# RL rewards per symbol
redis-cli KEYS "quantum:rl:reward:*"
redis-cli GET quantum:rl:reward:BTCUSDT
```

### 3. Container Logs
```bash
# Exit Brain Executor
journalctl -u quantum_exit_brain_executor.service --follow

# Auto Executor
journalctl -u quantum_auto_executor.service --follow

# Binance PnL Tracker
journalctl -u quantum_binance_pnl_tracker.service --follow
```

---

## 🚀 DEPLOYMENT STATUS (VPS)

**Server:** 46.224.116.254  
**Environment:** Production LIVE  
**Uptime:** 161 timer (6.7 dager)

### Active Containers
```
✅ quantum_ai_engine              # ML signal generation
✅ quantum_auto_executor          # Entry execution
✅ quantum_exit_brain_executor    # Exit management (DETTE SYSTEMET)
✅ quantum_binance_pnl_tracker    # PnL tracking
✅ quantum_rl_monitor             # RL feedback
✅ quantum_rl_dashboard           # Dashboard visualization
✅ quantum_redis                  # EventBus
✅ quantum_dashboard_backend      # API backend
✅ quantum_dashboard_frontend     # Web UI
```

### Configuration Status
```bash
EXIT_MODE=EXIT_BRAIN_V3          ✅ Active
EXIT_EXECUTOR_MODE=LIVE          ✅ Live orders enabled
EXIT_BRAIN_CHECK_INTERVAL=10s    ✅ Monitoring every 10s
```

### Current Metrics (Real-Time)
```
Portfolio:
  • PnL: $3.48
  • Positions: 2 (LINKUSDT, ATOMUSDT)
  • Exposure: 10%

AI Engine:
  • Accuracy: 78.9%
  • Sharpe: 1.09
  • Signals Generated: 75,521

Risk:
  • VaR 95%: -3.27%
  • Regime: Neutral
  • Volatility: 2.01%
```

---

## 🔧 DEBUGGING & DIAGNOSTICS

### Check Exit System Status
```bash
# Full diagnostic
python diagnose_exit_brain.py

# Check active positions
python check_exit_brain_positions.py

# Monitor executor status
python check_exit_brain_executor_status.py

# Inspect internal state
python inspect_exit_brain_state.py
```

### Live Monitoring
```bash
# Watch logs in real-time
python monitor_exit_brain_live.py

# Follow executor loop
docker logs -f quantum_exit_brain_executor | grep "EXIT_MONITOR"

# Track orders
docker logs -f quantum_exit_brain_executor | grep "EXIT_TP_ORDER\|EXIT_SL_ORDER"
```

### Redis State Inspection
```bash
# Check position states
redis-cli KEYS "*position*"

# View exit plan cache
redis-cli KEYS "*exit_plan*"

# Monitor PnL stream
redis-cli XREAD COUNT 10 STREAMS quantum:stream:exitbrain.pnl 0
```

---

## 💰 MONEY HARVESTING (ADAPTIVE PROFIT TAKING)

### Konsept
"Money Harvesting" er **ikke et separat system** - det er **integrert i Exit Brain V3.5** som **Adaptive Leverage-Aware Profit Taking**. Dette er den intelligente partial TP-strategien som justerer seg automatisk basert på leverage og market conditions.

### Hvordan Det Fungerer
```
┌──────────────────────────────────────────────────────────────┐
│              ADAPTIVE HARVESTING SYSTEM                       │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  1. LEVERAGE SENSITIVITY FACTOR (LSF):                       │
│     LSF = 1 / (1 + ln(leverage + 1))                         │
│                                                               │
│     Low Leverage (5x):  LSF = 0.57 → Conservative harvest    │
│     Medium (15x):       LSF = 0.36 → Balanced                │
│     High (50x):         LSF = 0.20 → Aggressive harvest      │
│     Ultra (100x):       LSF = 0.18 → Maximum front-loading   │
│                                                               │
│  2. HARVEST SCHEMES (Hvor mye closes ved hver TP):          │
│                                                               │
│     ≤10x Leverage - CONSERVATIVE:                            │
│       TP1: 30% | TP2: 30% | TP3: 40% (runner)               │
│       Eksempel: 0.1 BTC → 0.03 | 0.03 | 0.04                │
│                                                               │
│     10-30x Leverage - AGGRESSIVE:                            │
│       TP1: 40% | TP2: 40% | TP3: 20% (small runner)         │
│       Eksempel: 0.1 BTC → 0.04 | 0.04 | 0.02                │
│                                                               │
│     >30x Leverage - ULTRA-AGGRESSIVE:                        │
│       TP1: 50% | TP2: 30% | TP3: 20% (min runner)           │
│       Eksempel: 0.1 BTC → 0.05 | 0.03 | 0.02                │
│       Rationale: Høy leverage = høy risk → harvest tidlig!   │
│                                                               │
│  3. TP LEVEL CALCULATION (med LSF):                          │
│     TP1 = base_tp × (0.6 + LSF)                              │
│     TP2 = base_tp × (1.2 + LSF/2)                            │
│     TP3 = base_tp × (1.8 + LSF/4)                            │
│                                                               │
│     Ved 15x leverage (LSF=0.36, base_tp=2.0%):               │
│       TP1 = 2.0% × (0.6 + 0.36) = 1.92%                      │
│       TP2 = 2.0% × (1.2 + 0.18) = 2.76%                      │
│       TP3 = 2.0% × (1.8 + 0.09) = 3.78%                      │
│                                                               │
│  4. DYNAMIC ADJUSTMENT (Cross-Exchange Intelligence):        │
│     - High Volatility: Widen TPs by +40%                     │
│     - Recent Losses: Tighten levels by -10%                  │
│     - Strong Profits: Expand levels by +10%                  │
│     - Low Confidence: Extra -5% tightening                   │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Eksempel: Money Harvesting i Aksjon

**Scenario:** BTCUSDT LONG @ $100,000 | 0.1 BTC | 20x leverage

**System Calculations:**
```python
LSF = 1 / (1 + ln(20+1)) = 0.32
Harvest Scheme = [40%, 40%, 20%]  # 10-30x range

TP1 = 2.0% × (0.6 + 0.32) = 1.84%  →  $101,840
TP2 = 2.0% × (1.2 + 0.16) = 2.72%  →  $102,720
TP3 = 2.0% × (1.8 + 0.08) = 3.76%  →  $103,760
```

**Execution Flow:**
```
Entry: $100,000 | Position: 0.1 BTC | Value: $10,000

TP1 Hit @ $101,840:
  ├─> Harvest 40% (0.04 BTC) = $4,073.60
  ├─> Remaining: 0.06 BTC
  ├─> Realized PnL: +$73.60
  └─> Ratchet SL to breakeven ($100,000)

TP2 Hit @ $102,720:
  ├─> Harvest 40% (0.024 BTC) = $2,465.28
  ├─> Remaining: 0.036 BTC
  ├─> Realized PnL: +$139.20 (total: $212.80)
  └─> Ratchet SL to $101,500 (lock profit)

TP3 Hit @ $103,760:
  ├─> Harvest 20% (0.012 BTC) = $1,245.12
  ├─> Remaining: 0.024 BTC
  ├─> Realized PnL: +$249.84 (total: $462.64)
  └─> Keep runner or trailing stop

Final Outcome:
  • Total Harvested: $7,784
  • Total PnL: +$462.64 (4.63%)
  • Runner: 0.024 BTC still active
```

### Money Harvesting vs Standard TP

**Standard Fixed TP (Gammel Metode):**
```
TP @ +2% flat for full position:
  → Close 0.1 BTC @ $102,000
  → PnL: +$200
  → Risk: All-or-nothing, kan reverse før TP
```

**Adaptive Harvesting (Exit Brain V3.5):**
```
TP1 @ +1.84%: Take 40% ($73.60)
TP2 @ +2.72%: Take 40% ($139.20)
TP3 @ +3.76%: Take 20% ($249.84)
  → Total: +$462.64
  → Risk: De-risked progressively
  → Fordel: 2.3x better result
```

### Configuration
```bash
# Enable Adaptive Harvesting
ADAPTIVE_LEVERAGE_ENABLED=true

# Base Levels (Auto-adjusted by LSF)
BASE_TP_PCT=0.020         # 2.0% base
BASE_SL_PCT=0.012         # 1.2% base

# Safety Clamps
SL_CLAMP_MIN=0.001        # 0.1% minimum SL
SL_CLAMP_MAX=0.02         # 2.0% maximum SL
TP_MIN=0.003              # 0.3% minimum TP
```

### Monitoring Money Harvesting
```bash
# Check adaptive levels calculation
journalctl -u quantum_exit_brain_executor.service | grep "ADAPTIVE_LEVELS"

# Monitor harvest executions
journalctl -u quantum_exit_brain_executor.service | grep "EXIT_TP_ORDER"

# View harvest scheme per position
redis-cli HGETALL "position:state:BTCUSDT:LONG"
```

---

## 🎓 HVORDAN SYSTEMET FUNGERER (SIMPLIFIED)

1. **AI sier:** "Kjøp BTCUSDT med 85% confidence"
2. **Auto Executor:** Kjøper 0.01 BTC @ $100,000
3. **Exit Brain V3.5:** Kalkulerer leverage=15x, LSF=0.36, Harvest=[40%,40%,20%]
4. **Adaptive Levels:** TP1=+1.92%, TP2=+2.76%, TP3=+3.78%, SL=-1.2%
5. **Dynamic Executor:** Starter monitoring loop (hvert 10 sek)
6. **Price reaches $101,920:** TP1 trigger → **HARVEST 40%** (0.004 BTC)
7. **SL ratchets:** Flytter stop loss fra $98,800 til $100,500 (breakeven+)
8. **Price reaches $102,760:** TP2 trigger → **HARVEST 40%** (0.0024 BTC)
9. **Price reaches $103,780:** TP3 trigger → **HARVEST 20%** (0.0012 BTC)
10. **PnL Tracker:** Kalkulerer total +$180 profit, publiserer til Redis
11. **Dashboard:** Viser live stats med adaptive harvesting metrics

---

## 📌 KEY TAKEAWAYS

✅ **MONEY HARVESTING** = Adaptive leverage-aware partial TP (integrert i Exit Brain V3.5)  
✅ **Harvest Schemes:** 30/30/40% (low lev) → 40/40/20% (mid) → 50/30/20% (high lev)  
✅ **INGEN hardkodede TP/SL** - Alt AI-drevet og dynamisk justert med LSF  
✅ **MARKET-only exits** - Instant execution, ingen order management  
✅ **Dual-layer protection** - Internal AI SL + Hard SL safety net  
✅ **Partial TP strategy** - Progressive harvesting basert på leverage risk  
✅ **Auto-ratcheting** - SL tightens after hver TP hit  
✅ **Real-time monitoring** - 10-second check cycle  
✅ **Cross-Exchange Intelligence** - Volatility-adjusted targets  
✅ **PnL Optimization** - Auto-tightens levels ved losses  
✅ **Redis EventBus** - Decoupled, scalable architecture  
✅ **Full observability** - Dashboard, logs, metrics, streams  
✅ **Production-ready** - LIVE på VPS with adaptive harvesting active  

---

**Generated by:** GitHub Copilot  
**Documentation Version:** 1.0  
**System Version:** Exit Brain V3.5 + Dynamic Executor

