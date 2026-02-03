# RUNBOOK: LIVE OPERATIONS

**Document Version**: 1.0  
**Last Updated**: 2026-01-02  
**Operator**: Quantum Trader Team  

---

## OVERVIEW

This runbook provides operational procedures for managing the Quantum Trader system in **LIVE PRODUCTION** mode with real trading.

**Phases**:
- Phase A: Preflight (verification only)
- Phase B: Shadow Mode (MAINNET data, paper execution)
- Phase C: Live Small (real trading, micro-notional)
- Phase D: Scale-Up (gradual increase)
- Emergency: Abort (kill-switch)

---

## PHASE A: PREFLIGHT VERIFICATION

**Purpose**: Verify all systems before any trading activity.

**Command**:
```bash
cd /mnt/c/quantum_trader
wsl bash scripts/go_live_preflight.sh
```

**Duration**: ~5 minutes

**What It Checks**:
1. P1-B prerequisites (0 unhealthy containers, Prometheus targets UP)
2. Mode flags (testnet/paper/live verification)
3. Binance connectivity (serverTime, exchangeInfo, balance)
4. Redis streams (intent queue health)
5. ESS/kill-switch (circuit breaker config)
6. Observability (Prometheus, Grafana, alerts)
7. Disk headroom (>20% free)

**Acceptance**:
- ✅ ALL checks PASS → Proceed to Phase B
- ❌ ANY check FAIL → Fix issues, re-run preflight

**Output**: `GO_LIVE_PREFLIGHT_PROOF.md`

---

## PHASE B: SHADOW MODE

**Purpose**: Run full pipeline against MAINNET data without sending real orders.

**Prerequisites**:
- Phase A passed
- Environment configured:
  ```
  BINANCE_USE_TESTNET=false
  PAPER_TRADING=true
  ```

**Command**:
```bash
wsl bash scripts/go_live_shadow.sh
```

**Duration**: 60 minutes (configurable)

**What It Does**:
1. Verifies shadow mode configuration
2. Monitors intent processing
3. Verifies NO orders submitted (PAPER_TRADING=true)
4. Checks for "WOULD_SUBMIT" log entries
5. Captures baseline vs final metrics

**Acceptance**:
- ✅ Intents processed >0
- ✅ Orders submitted = 0
- ✅ WOULD_SUBMIT logs >0
- ❌ ANY order leakage → FAIL

**Output**: `GO_LIVE_SHADOW_PROOF.md`

---

## PHASE C: LIVE SMALL 🚨

**Purpose**: Execute 1-3 real orders with extreme risk controls.

**Prerequisites**:
- Phase A and B passed
- Environment configured:
  ```
  PAPER_TRADING=false
  BINANCE_USE_TESTNET=false
  SYMBOL_ALLOWLIST=BTCUSDT,ETHUSDT (max 3)
  MAX_POSITIONS=1
  MAX_LEVERAGE=2
  MAX_NOTIONAL_PER_TRADE=50 (USDT)
  COOLDOWN_SECONDS=60
  ```

**⚠️ WARNING**: This phase uses REAL MONEY.

**Command**:
```bash
wsl bash scripts/go_live_live_small.sh
# (requires typing 'LIVE' to confirm)
```

**Duration**: 120 minutes (or until 3 orders executed)

**What It Does**:
1. Verifies live configuration
2. Captures baseline order count
3. Monitors order execution
4. Captures order proofs (orderId, logs)
5. Checks for critical errors (-4045, -1111)
6. Verifies position creation and TP/SL placement

**Acceptance**:
- ✅ 1-3 orders executed
- ✅ Order proofs captured (orderId visible)
- ✅ No critical errors
- ✅ Positions created + TP/SL active
- ❌ Order loops or balance errors → FAIL

**Output**: `GO_LIVE_LIVE_SMALL_PROOF.md`

**Monitoring During Live Small**:
```bash
# Watch executor logs
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 "docker logs -f quantum_auto_executor"

# Check orders
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 "redis-cli GET quantum:counter:orders_submitted"

# Check positions
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 "redis-cli HGETALL quantum:positions:open"
```

---

## PHASE D: GRADUAL SCALE-UP

**Prerequisites**:
- Phase C passed
- 24h stability observed (no crashes, no loops, PnL reasonable)

**Process**:
1. **Day 1**: Live Small (1-3 symbols, 1 position, 2x leverage)
2. **Day 2**: Increase symbols to 5-7, max positions to 2
3. **Day 3**: Increase leverage to 3x (if PnL positive)
4. **Week 2**: Increase max notional per trade
5. **Week 3**: Full symbol universe (if stable)

**Change Procedure**:
1. Update environment variables in `.env.vps`
2. Restart services: `docker compose up -d auto_executor`
3. Monitor for 2-4 hours
4. Document change in `GO_LIVE_CHANGELOG.md`

**Rollback If**:
- Order loops appear
- Drawdown >5% in single session
- Critical errors (-4045, -1111, -2019)
- Unexpected behavior

---

## EMERGENCY: ABORT / KILL-SWITCH 🚨

**When to Use**:
- Runaway order loops
- Unexpected high drawdown
- Exchange connectivity issues
- System behaving erratically
- Operator discretion

**Command**:
```bash
wsl bash scripts/go_live_abort.sh
# (requires typing 'ABORT' to confirm)
```

**What It Does**:
1. Sets emergency feature flags (disables new entries)
2. Cancels all open orders
3. Optionally closes all positions (operator choice)
4. Stops auto_executor and ai_engine containers

**Post-Abort**:
- System enters EMERGENCY STOP mode
- Trading halted until manual recovery
- Investigate root cause
- Document incident

**Recovery**:
```bash
# On VPS
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254

# Clear emergency flags
redis-cli DEL quantum:config:emergency_stop quantum:feature:new_entries_disabled

# Restart services
cd /home/qt/quantum_trader
docker compose up -d auto_executor ai_engine

# Re-run preflight
wsl bash /mnt/c/quantum_trader/scripts/go_live_preflight.sh
```

**Output**: `GO_LIVE_ABORT_PROOF.md`

---

## MONITORING DASHBOARDS

**Grafana**: http://46.224.116.254:3000
- Dashboard: "Quantum Trader Overview"
- Dashboard: "Trading Execution Metrics"
- Dashboard: "Risk & Performance"

**Key Metrics**:
- Orders submitted (counter)
- Positions open (gauge)
- PnL (cumulative)
- Order latency (histogram)
- Execution blocks (counter)
- Risk violations (counter)

**Prometheus Alerts**:
- AIEngineDown
- AutoExecutorDown
- HighOrderLatency
- RiskViolationDetected
- DiskUsageHigh

---

## DAILY OPERATIONS CHECKLIST

### Morning Check (09:00 UTC)
- [ ] Check Grafana for alerts
- [ ] Verify container health: `systemctl list-units --filter health=unhealthy`
- [ ] Check disk usage: `df -h | grep sda1`
- [ ] Review overnight PnL
- [ ] Check for error spikes in logs

### Midday Check (15:00 UTC)
- [ ] Monitor open positions count
- [ ] Check order execution rate
- [ ] Verify no stuck intents in queue
- [ ] Review TP/SL placement accuracy

### Evening Check (21:00 UTC)
- [ ] Daily PnL review
- [ ] Check for any new alerts
- [ ] Review order success rate
- [ ] Plan next day adjustments (if needed)

---

## TROUBLESHOOTING

### Issue: No Orders Being Submitted

**Check**:
1. Emergency stop flag: `redis-cli GET quantum:config:emergency_stop`
2. Intent queue: `redis-cli XLEN quantum:stream:intent`
3. Executor logs: `journalctl -u quantum_auto_executor.service --tail 100`
4. Balance check: Executor logs should show available balance

**Fix**:
- If emergency_stop=true: Clear flag and restart executor
- If no intents: Check AI engine is running and publishing signals
- If balance insufficient: Check Binance account funding

---

### Issue: Order Loops (-4045 Errors)

**Symptoms**: Repeated order attempts with -4045 (reduce-only violation)

**Immediate Action**:
```bash
wsl bash scripts/go_live_abort.sh
```

**Investigation**:
1. Check position side vs order side mismatch
2. Review EXIT_BRAIN logs for TP/SL placement
3. Verify positionSide parameter in orders

**Fix**: Redeploy with positionSide fix (already patched in v3.5)

---

### Issue: High Drawdown

**Threshold**: >3% in single session

**Action**:
1. Review recent trades in Grafana
2. Check if specific symbol or strategy causing loss
3. Consider reducing leverage or symbol allowlist
4. If >5% drawdown: Execute abort script
5. Analyze root cause before resuming

---

## AI UNIVERSE VERIFICATION

**Purpose**: Verify guardrails pipeline ran successfully on last policy refresh

**Command**:
```bash
# Verify guardrails executed in last 2 hours
journalctl -u quantum-policy-refresh.service --since "2 hours ago" --no-pager | \
  grep -E "AI_UNIVERSE_GUARDRAILS|AI_UNIVERSE_PICK" | tail -30

# Expected output:
# - 1 line: AI_UNIVERSE_GUARDRAILS total=540 vol_ok=111 spread_checked=80 ... vol_src=quoteVolume
# - 10 lines: AI_UNIVERSE_PICK symbol=XXX qv24h_usdt=NNN spread_bps=X.X ...
```

**What to Check**:
- `vol_ok` count (should be ~100-120 after $20M filter)
- `spread_checked` (should be ≤80 for optimization)
- `spread_skipped` (should be >0, confirms top-N optimization)
- `excluded_vol` (should be ~400-450, confirms microcap filtering)
- `vol_src=quoteVolume` (confirms using USDT volume, not base)

**Red Flags**:
- No AI_UNIVERSE_GUARDRAILS log → generator failed or didn't run
- `vol_ok=0` → volume filter too aggressive
- `vol_ok=540` → volume filter not working
- `spread_checked=111` → optimization not active
- Missing `vol_src=quoteVolume` → using wrong volume source
- **`age_ok < 10` in >2 consecutive refreshes → DEGRADED UNIVERSE**
  - Action: Lower thresholds temporarily OR increase MAX_SPREAD_CHECKS
  - Reason: Trading on too few symbols increases concentration risk
  - Example: `MIN_QUOTE_VOL_USDT_24H=10000000` (reduce from 20M to 10M)
  - Example: `MAX_SPREAD_CHECKS=120` (increase from 80 to 120)
- Missing `vol_src=quoteVolume` → using wrong volume source

**Manual Trigger + Full Verification** (instant proof that everything works):
```bash
# Trigger policy refresh + verify metadata + guardrails (one command)
systemctl start quantum-policy-refresh.service \
  && sleep 5 \
  && redis-cli HMGET quantum:policy:current generator policy_version market stats_endpoint universe_hash features_window \
  && journalctl -u quantum-policy-refresh.service --since "5 minutes ago" --no-pager | egrep "AI_UNIVERSE_GUARDRAILS|AI_UNIVERSE_PICK|POLICY_"
```

**Expected Output:**
```
# Redis metadata (6 lines):
generator: ai_universe_v1
policy_version: 1.0.0-ai-v1
market: futures
stats_endpoint: fapi/v1/ticker/24hr
universe_hash: e03793cd5579a5a2
features_window: 15m,1h

# Guardrails log (1 line):
AI_UNIVERSE_GUARDRAILS total=540 vol_ok=111 ... metadata_ok=1 missing_fields=""

# PICK logs (10 lines):
AI_UNIVERSE_PICK symbol=RIVERUSDT ... spread_detail_missing=0

# Policy saved:
POLICY_SAVED: version=1.0.0-ai-v1 hash=6e4f9363
```

**Alternative** (dry-run without saving policy):
```bash
cd /root/quantum_trader
python3 scripts/ai_universe_generator_v1.py --dry-run  # Test only
```

---

## CONTACTS

**VPS**: 46.224.116.254 (Hetzner)  
**SSH Key**: `~/.ssh/hetzner_fresh`  
**Git Repo**: https://github.com/binyaminsemerci-ops/quantum_trader  

**Emergency Contacts**:
- Primary Operator: [Your contact]
- Backup Operator: [Backup contact]

---

## CHANGE LOG

| Date | Change | Operator | Result |
|------|--------|----------|--------|
| 2026-01-02 | Phase A Preflight | Ops Team | PASS ✅ |
| TBD | Phase B Shadow | TBD | TBD |
| TBD | Phase C Live Small | TBD | TBD |

*(Update this table in GO_LIVE_CHANGELOG.md as you progress)*

---

**FINAL REMINDER**: Live trading involves real financial risk. Always start with micro-notional, monitor closely, and use the abort script if anything looks wrong.

