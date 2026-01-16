# GO-LIVE CHECKLIST: TESTNET → MAINNET

**Date**: _______________  
**Operator**: _______________  
**Target**: Production MAINNET  
**Status**: 🔴 NOT STARTED

---

## ⚠️ CRITICAL GATES (MUST PASS BEFORE PROCEEDING)

### Gate 0: P1-B Prerequisites ✅ REQUIRED
- [ ] All containers healthy (`systemctl list-units --filter health=unhealthy` = empty)
- [ ] Disk usage <85% (current: ___%)
- [ ] Prometheus targets UP (check `/api/v1/targets`)
- [ ] Alert rules loaded (check `/api/v1/rules`)
- [ ] No critical alerts firing

**If ANY Gate 0 item fails: DO NOT PROCEED. Fix P1-B first.**

---

## 🎯 PHASE A: PREFLIGHT VERIFICATION (No Trading)

**Script**: `scripts/go_live_preflight.sh`  
**Duration**: 5-10 minutes  
**Risk Level**: 🟢 ZERO (no trading activity)

### Checklist
- [ ] **Mode Flags Verified**
  - [ ] BINANCE_USE_TESTNET value confirmed
  - [ ] PAPER_TRADING value confirmed
  - [ ] LIVE_TRADING_ENABLED value confirmed

- [ ] **Binance MAINNET Connectivity**
  - [ ] API keys valid (not testnet keys)
  - [ ] `/api/v3/time` responds (<500ms latency)
  - [ ] `/fapi/v1/exchangeInfo` responds
  - [ ] Account balance readable
  - [ ] No rate limit warnings

- [ ] **Redis Streams Healthy**
  - [ ] `quantum:stream:intent` exists
  - [ ] `quantum:stream:execution` exists
  - [ ] Consumer groups configured
  - [ ] No backlog >100 messages

- [ ] **ESS/Kill-Switch Pathways**
  - [ ] ESS config exists (`/config/risk_config.json`)
  - [ ] Kill-switch endpoint responds (`/api/kill-switch/status`)
  - [ ] Circuit breaker config loaded

- [ ] **Observability Ready**
  - [ ] Prometheus targets UP (critical services only)
  - [ ] Grafana dashboards load
  - [ ] Alert rules loaded (>10 rules)
  - [ ] Alertmanager routes configured

- [ ] **Resource Headroom**
  - [ ] Disk <80%
  - [ ] Memory <70%
  - [ ] CPU <50% average

**Output**: `GO_LIVE_PREFLIGHT_PROOF.md`  
**Acceptance**: ✅ ALL checks PASS, proof document generated

**If ANY check fails**: 🛑 STOP. Fix issue and re-run preflight.

---

## 🕵️ PHASE B: SHADOW MODE (Live Data, Paper Execution)

**Script**: `scripts/go_live_shadow.sh`  
**Duration**: 30-60 minutes  
**Risk Level**: 🟡 LOW (no real orders, but uses MAINNET keys/data)

### Configuration
```bash
BINANCE_USE_TESTNET=false
PAPER_TRADING=true
LIVE_TRADING_ENABLED=false
SHADOW_MODE=true
```

### Checklist
- [ ] **Configuration Applied**
  - [ ] Testnet mode disabled
  - [ ] Paper trading enabled
  - [ ] Live trading disabled
  - [ ] Services restarted with new config

- [ ] **Data Flow Verified**
  - [ ] Market data from MAINNET (verify symbol prices)
  - [ ] AI signals generated from MAINNET data
  - [ ] Intents published to Redis
  - [ ] Executor consumes intents

- [ ] **Paper Execution Logs**
  - [ ] `INTENT_RECEIVED` counter increases
  - [ ] `ORDER_SUBMIT` counter = 0 (no real orders)
  - [ ] `WOULD_SUBMIT` logs appear (dry-run behavior)
  - [ ] TP/SL calculated but not placed

- [ ] **No Errors/Blocks**
  - [ ] No `-4045` errors (insufficient balance)
  - [ ] No `-1111` errors (precision issues)
  - [ ] No rate limit hits
  - [ ] No execution blocks from ESS

- [ ] **Metrics Validation**
  - [ ] `intent_received_total` > 0
  - [ ] `order_submitted_total` = 0
  - [ ] `execution_blocks_total` = 0 (or expected ESS blocks)

**Duration**: Run for 30-60 minutes, observe logs continuously

**Output**: `GO_LIVE_SHADOW_PROOF.md`  
**Acceptance**: ✅ No errors, intents flow correctly, no real orders placed

**If shadow mode fails**: 🛑 STOP. Investigate logs, fix issues, re-run shadow.

---

## 🚀 PHASE C: LIVE SMALL (Micro-Notional, Hard Caps)

**Script**: `scripts/go_live_live_small.sh`  
**Duration**: 2-4 hours (manual monitoring)  
**Risk Level**: 🟠 MEDIUM (real orders, but micro-notional)

### Configuration (ULTRA-CONSERVATIVE)
```bash
BINANCE_USE_TESTNET=false
PAPER_TRADING=false
LIVE_TRADING_ENABLED=true

# HARD CAPS (CRITICAL!)
ALLOWED_SYMBOLS="BTCUSDT,ETHUSDT"  # Max 2-3 symbols
MAX_OPEN_POSITIONS=1
MAX_LEVERAGE=2
MAX_NOTIONAL_PER_TRADE=20  # $20 USDT max
MIN_ENTRY_COOLDOWN=60  # 60 seconds between entries
MAX_TRADES_PER_HOUR=5

# KILL-SWITCH ARMED
ENABLE_KILL_SWITCH=true
AUTO_STOP_ON_LOSS_PERCENT=5  # Stop if -5% account loss
```

### Pre-Live Checklist
- [ ] **Risk Caps Verified**
  - [ ] Symbol allowlist active (only 2-3 symbols)
  - [ ] Max positions = 1
  - [ ] Max leverage = 2x
  - [ ] Max notional = $20 per trade
  - [ ] Entry cooldown = 60s minimum

- [ ] **Account Balance**
  - [ ] Sufficient USDT in MAINNET account (min $100)
  - [ ] Account NOT over-allocated
  - [ ] No existing open positions

- [ ] **Kill-Switch Ready**
  - [ ] Kill-switch endpoint accessible
  - [ ] Abort script tested (dry-run)
  - [ ] Telegram/Slack alerts configured

- [ ] **Monitoring Active**
  - [ ] Grafana dashboard open (live view)
  - [ ] Logs tailing in terminal
  - [ ] Alertmanager ready to notify

### Live Execution Checklist
- [ ] **First Trade Proof (within 30 minutes)**
  - [ ] Intent generated and published
  - [ ] Executor consumed intent
  - [ ] Order submitted to Binance (log: `ORDER_SUBMIT`)
  - [ ] Order response received (log: `ORDER_RESPONSE`, orderId present)
  - [ ] Position opened (verify in Binance UI)
  - [ ] Stop-loss placed (verify orderId)
  - [ ] Take-profit placed (verify orderId)

- [ ] **No Critical Errors**
  - [ ] No `-4045` (insufficient balance)
  - [ ] No `-1111` (precision errors)
  - [ ] No position side conflicts
  - [ ] No TP/SL placement failures

- [ ] **Observability Proof**
  - [ ] Prometheus metrics update (`order_submitted_total` > 0)
  - [ ] Grafana shows position open
  - [ ] No unexpected alerts firing

- [ ] **Exit Proof (when exit signal triggers)**
  - [ ] Exit brain generated exit decision
  - [ ] Market order placed to close position
  - [ ] Position closed (verify balance change)
  - [ ] PnL recorded

**Acceptance**: ✅ 1-3 successful trades with full proof (entry, TP/SL, exit)

**Duration**: Monitor for 2-4 hours. If 1-3 trades complete successfully → PASS

**Output**: `GO_LIVE_LIVE_SMALL_PROOF.md`

**If ANY critical error occurs**: 🛑 RUN ABORT SCRIPT (`go_live_abort.sh`)

---

## 📈 PHASE D: GRADUAL SCALE-UP (Controlled Expansion)

**Prerequisites**: 24+ hours stable operation in LIVE SMALL mode

### Scale-Up Steps (One per day minimum)

#### Day 1 → Day 2
- [ ] Increase symbol allowlist: 2-3 → 5 symbols
- [ ] Max positions: 1 → 2
- [ ] Max notional: $20 → $50
- [ ] Document in `GO_LIVE_CHANGELOG.md`

#### Day 2 → Day 3
- [ ] Symbol allowlist: 5 → 10 symbols
- [ ] Max positions: 2 → 3
- [ ] Max notional: $50 → $100
- [ ] Document in `GO_LIVE_CHANGELOG.md`

#### Day 3 → Week 1
- [ ] Max leverage: 2x → 5x (if proven stable)
- [ ] Symbol allowlist: 10 → universe (if risk permits)
- [ ] Max positions: 3 → 5
- [ ] Max notional: $100 → $200
- [ ] Document in `GO_LIVE_CHANGELOG.md`

### Scale-Up Checklist (Before Each Increase)
- [ ] No critical incidents in past 24h
- [ ] Win rate >40%
- [ ] Max drawdown <10%
- [ ] No ESS kill-switch triggers
- [ ] All alerts resolved
- [ ] Operator approval

**Rule**: If ANY metric degrades → PAUSE scale-up, investigate

---

## 🔴 ABORT PROCEDURE (Emergency Kill-Switch)

**Trigger Conditions** (any of these):
- Critical error (e.g., -4045 loop, precision errors)
- Unexpected loss >5% account
- Position side conflicts
- TP/SL placement failures
- Rate limit hits
- Operator decision

**Abort Script**: `scripts/go_live_abort.sh`

### Abort Checklist
- [ ] Run: `scripts/go_live_abort.sh`
- [ ] Verify: New entries stopped
- [ ] Verify: Open orders cancelled
- [ ] Verify: Positions closed (if policy enabled)
- [ ] Verify: Services reverted to TESTNET/PAPER mode
- [ ] Document: `GO_LIVE_ABORT_PROOF.md`

**Acceptance**: System returns to safe state (paper/testnet)

---

## 📋 FINAL SIGN-OFF

### Phase Completion
- [ ] Phase A: Preflight ✅
- [ ] Phase B: Shadow ✅
- [ ] Phase C: Live Small ✅
- [ ] Phase D: Scale-Up (in progress)

### Operator Notes
```
Date: _______________
Phase: _______________
Status: _______________
Notes: _______________
```

### Rollback Readiness
- [ ] Abort script tested
- [ ] Rollback plan documented (`GO_LIVE_ROLLBACK.md`)
- [ ] Operator knows how to stop live trading (1 command)

---

## 🎯 SUCCESS CRITERIA

### Phase A: Preflight
- ✅ All checks pass, proof document generated

### Phase B: Shadow
- ✅ No errors, intents flow, no real orders placed

### Phase C: Live Small
- ✅ 1-3 trades complete with full proof (entry, TP/SL, exit)
- ✅ No critical errors
- ✅ Observability working

### Phase D: Scale-Up
- ✅ 24h+ stability at each level
- ✅ Metrics healthy, no degradation

---

**REMEMBER**: You can ALWAYS abort. Better to stop and investigate than risk capital.

**Operator**: If in doubt, run `go_live_abort.sh` and revert to testnet/paper mode.

