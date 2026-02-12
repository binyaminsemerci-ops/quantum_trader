# 📐 PnL AUTHORITY ESCALATION RULEBOOK (v1.0)

**System**: quantum_trader  
**Purpose**: Definere eksakte, målbare, ikke-forhandlingsbare beviskrav for autoritetsnivå  
**Authority**: Canonical reference for all PnL authority changes  
**Version**: 1.0  
**Effective Date**: February 10, 2026  

---

## 🎯 GRUNNPRINSIPP

**Autoritet gis ikke fordi noe er smart.**  
**Den gis fordi systemet blir bedre når det er på.**

Hvis du ikke kan bevise det → forblir komponenten på sitt nivå.

---

## 🔄 TILLATTE OVERGANGER

```
🔴 DEAD → ⚪ OBSERVER → 🔵 SCORER → 🟡 GATEKEEPER → 🟢 CONTROLLER
```

### ⚠️ STRENGE REGLER

1. **Kun én nivå om gangen** (ingen hopp: DEAD → SCORER ❌)
2. **Alle beviskrav må være oppfylt** (ikke "nesten alle")
3. **Produksjonsbevis kun** (ingen lab/test/synthetic data)
4. **Nedgradering umiddelbart** hvis bevis forsvinner
5. **Ingen "pilot"** eller "vi tror" - kun målbar runtime-sannhet

---

## 🔴 → ⚪ DEAD → OBSERVER

**Definisjon**: "Denne komponenten eksisterer faktisk i runtime."

### ✅ MINIMUM BEVISKRAV

Alle må være oppfylt i produksjon:

#### 1. Service Liveness
```bash
systemctl is-active <service> == active
```
- Kjører kontinuerlig
- Ikke crashlooping
- PID stable i minst 1 time

#### 2. Kontinuerlig Output
- Redis key / stream / file **vokser over tid**
- Ikke bare ved oppstart
- Minimum 3 nye events per forventet intervall
- Eksempel:
  ```bash
  # If cadence = 1 min, require:
  redis-cli XLEN quantum:stream:<output> increases every 1 min
  ```

#### 3. Tidsnærhet
- Ny output ≤ **5× forventet intervall**
  - Eksempel: Hvis cadence = 1 min → latest event < 5 min gammel
  - Eksempel: Hvis cadence = 1 time → latest event < 5 timer gammel
- Timestamp i payload må være innenfor tolerance
- Ikke stale data recycling

#### 4. Deserialiserbarhet
- Payload kan dekodes **uten fallback / empty dict**
- Felt valideres:
  - `symbol` ≠ null
  - `timestamp` ≠ null  
  - `value` ≠ null / NaN
- JSON parsing succeeds 100% (no TypeError/KeyError i logs)

### 🧪 VERIFICATION SCRIPT

```bash
# Template for OBSERVER-level verification:

# 1. Liveness check
systemctl is-active quantum-<component>.service

# 2. Output growth check
redis-cli XLEN quantum:stream:<output>
# Wait 5 min
redis-cli XLEN quantum:stream:<output>
# Verify: count increased

# 3. Recency check
redis-cli XREVRANGE quantum:stream:<output> + - COUNT 1
# Parse timestamp, verify < 5× expected interval

# 4. Deserializability check
redis-cli XREVRANGE quantum:stream:<output> + - COUNT 10 | \
  python3 -c "import json,sys; [json.loads(line) for line in sys.stdin]"
# Verify: no exceptions, all required fields present
```

### 🚫 DISKVALIFISERING

Komponenten faller tilbake til **DEAD** hvis:
- Output er tom (XLEN = 0 eller zero file size)
- Kun logging (ingen structured data output)
- Kun startup-write (ingen updates etter init)
- Deserialiseringsfeil > 5% av payloads
- Stale data > 5× expected interval

### 🎯 RESULTAT

➡️ **Komponenten kan observeres, men har null autoritet**

Eksempler:
- CLM samler trade.closed events
- RL Trainer produserer metrics
- AI Exit Evaluator skriver decisions (selv om ingen leser)

---

## ⚪ → 🔵 OBSERVER → SCORER

**Definisjon**: "Denne komponenten påvirker beslutninger, men kan ikke stoppe dem."

### ✅ BEVISKRAV (ALLE)

#### 1. Konsumert Output

**Minimum**: Minst én annen runtime-komponent **leser verdien**

**Bevis (velg minst én)**:
```bash
# A) Redis Stream consumer:
redis-cli XINFO GROUPS quantum:stream:<output> | grep -A 5 "name.*<consumer>"
# Verify: lag < 100, pending messages processed

# B) Code path verification:
grep -r "quantum:stream:<output>\|<output_key>" /root/quantum_trader/microservices/<consumer> --include="*.py"
# Verify: actual .get() or XREAD calls, not just comments

# C) Execution trace:
journalctl -u quantum-<consumer>.service --since "1 hour ago" | grep "<component_name>\|<output_key>"
# Verify: logs show actual value usage
```

**Krav**:
- Ikke bare logging av verdien
- Faktisk brukt i downstream logic
- Consumer må være **OBSERVER-level** eller høyere selv

#### 2. Beslutningskontakt

**Minimum**: Verdien brukes i minst én av:
- **Threshold**: `if score > X then ACTION`
- **Confidence**: `final_confidence = base * scorer_weight`
- **Ranking**: `sorted_candidates = sort_by(scorer_output)`
- **Filtering**: `if scorer_ok: proceed else: next_candidate`

**Bevis**:
```python
# Example code patterns that qualify:

# VALID - Threshold:
if exit_evaluator.score > 0.7:
    execute_exit()

# VALID - Confidence weighting:
final_confidence = base_confidence * (1 + clm_adjustment)

# VALID - Ranking:
candidates = sorted(signals, key=lambda x: x.ml_score, reverse=True)

# INVALID - Pure logging:
logger.info(f"AI score: {ai_score}")  # ❌ No decision impact

# INVALID - Conditional logging:
if ai_score > 0.5:
    logger.debug("High confidence detected")  # ❌ No execution impact
```

**Diskvalifisering**: Hvis scorer kun brukes til:
- Logging / telemetri
- Visualisering
- Debug output
- Metrics som ikke påvirker execution

#### 3. Ikke-Autoritær

**Krav**: Systemet **kan handle uten komponenten**

**Bevis**:
```python
# VALID - Explicit fallback:
if scorer_available():
    confidence = scorer.get_confidence()
else:
    confidence = DEFAULT_CONFIDENCE  # ← Explicit fallback

# VALID - Optional weighting:
score = base_score * (1.0 + optional_scorer_adjustment())

# INVALID - Hard dependency:
score = scorer.get_score()  # ← No fallback, crashes if scorer fails ❌
```

**Fallback må være**:
- Eksplisitt dokumentert i kode (kommentar eller docstring)
- Deterministisk (ikke random)
- Konservativt (bias mot safety)

#### 4. Reproduserbar Effekt

**Krav**: Samme input → samme score → samme beslutning

**Bevis**: Unit test eller integration test:
```python
def test_scorer_determinism():
    input_state = {...}
    
    # Run scorer twice
    score_1 = scorer.evaluate(input_state)
    score_2 = scorer.evaluate(input_state)
    
    assert score_1 == score_2  # Deterministic
    
    # Verify decision impact
    decision_1 = decider.decide(score_1)
    decision_2 = decider.decide(score_2)
    
    assert decision_1 == decision_2
```

**Diskvalifisering**: Hvis scorer har:
- Random components uten seed
- Tidsavhengige decisions (utover timestamp i input)
- External API calls med variable results

#### 5. Isolert Påvirkning

**Krav**: Scorer påvirker **KUN ÉN** beslutningsdimensjon

**Tillatt** (velg én):
- Entry confidence (0.0 - 1.0)
- Exit timing score (-1.0 to 1.0)
- Position sizing multiplier (0.5x - 2.0x)
- Symbol ranking (ordinal)

**FORBUDT**: Scorer som påvirker flere dimensjoner samtidig
```python
# INVALID - Multiple dimensions:
class BadScorer:
    def evaluate(self):
        return {
            "entry_confidence": 0.8,    # ← Dimension 1
            "position_size": 150.0,      # ← Dimension 2 ❌
            "exit_threshold": 0.02       # ← Dimension 3 ❌
        }

# VALID - Single dimension:
class GoodScorer:
    def evaluate(self):
        return {
            "entry_confidence": 0.8,  # ← Only one decision dimension
            "metadata": {...}          # ← Metadata OK (not used in decisions)
        }
```

### 🧪 VERIFICATION SCRIPT

```bash
# Template for SCORER-level verification:

# 1. Consumer verification
redis-cli XINFO GROUPS quantum:stream:<scorer_output> | grep "pending\|lag"
# Verify: lag < 100, pending < 50

# 2. Decision contact verification
grep -B 5 -A 10 "if.*<scorer_output>\|<scorer_key>" \
  /root/quantum_trader/microservices/<consumer>/main.py
# Verify: actual branching or math using scorer value

# 3. Fallback verification
grep -B 5 -A 5 "else.*DEFAULT\|fallback\|scorer.*None" \
  /root/quantum_trader/microservices/<consumer>/main.py
# Verify: explicit fallback path exists

# 4. Determinism verification
# Run component twice with same input, compare outputs:
echo '{"symbol": "BTCUSDT", "price": 50000}' | \
  python3 -m microservices.<scorer>.main
# Verify: identical output on repeat

# 5. Isolation verification
redis-cli XREVRANGE quantum:stream:<scorer_output> + - COUNT 1
# Parse output, verify only ONE decision dimension present
```

### 🚫 DISKVALIFISERING

Komponenten faller tilbake til **OBSERVER** hvis:
- Output ignoreres (lag > 1000 eller no consumers)
- Fallback er "best guess" eller stochastic
- Flere beslutningsdimensioner påvirkes
- Non-deterministic behavior uten explicit randomness control
- Consumer crashes når scorer unavailable

### 🎯 RESULTAT

➡️ **Komponenten får begrenset innflytelse, men ingen veto**

Eksempler:
- Exit evaluator score brukes til å **rangere kandidater** (ikke blokkere)
- CLM adjustment brukes til **confidence weighting** (ikke entry blocking)
- Regime detector brukes til **threshold adjustment** (ikke hard gate)

---

## 🔵 → 🟡 SCORER → GATEKEEPER

**Definisjon**: "Denne komponenten kan stoppe handlinger."

### ✅ BEVISKRAV (EKSTREMT STRIKT)

#### 1. Hard Blokkering

**Krav**: Eksplisitt **return / abort / executed=False**

**Bevis** (code inspection):
```python
# VALID - Explicit blocking:
if not gatekeeper.check(symbol):
    logger.info(f"Gatekeeper BLOCKED {symbol}")
    return False  # ← Execution stops here ✅

# VALID - Exception-based:
try:
    gatekeeper.validate(order)
except GatekeeperVeto as e:
    logger.error(f"Order rejected: {e}")
    return False  # ← Execution stops ✅

# INVALID - Soft ignore:
if not gatekeeper.check(symbol):
    logger.warning(f"Gatekeeper concern: {symbol}")
    # continues anyway ❌
```

**Diskvalifisering**: Hvis gatekeeper:
- Logs warnings but allows execution
- Soft-reject (continues on negative check)
- Advisory-only (result ignored by consumer)

#### 2. Deterministisk Regel

**Krav**: Samme input → **alltid** samme blokkering

**Bevis**: Integration test med 100+ samples:
```python
def test_gatekeeper_determinism():
    test_cases = [
        {"symbol": "ZZZUSDT", "expected_result": False},  # Not in universe
        {"symbol": "BTCUSDT", "expected_result": True},   # In universe
        {"symbol": "ETHUSDT", "expected_result": True},   # In universe
    ]
    
    for case in test_cases:
        for _ in range(100):  # Test 100 times
            result = gatekeeper.check(case["symbol"])
            assert result == case["expected_result"]
```

**Diskvalifisering**: Hvis gatekeeper har:
- Stochastic components (random rejections)
- ML models uten threshold (soft predictions)
- Time-dependent logic (except explicit time-based rules)

#### 3. Audit Trail

**Krav**: Hver blokkering logges med:
- **Årsak** (konkret regel som trigget)
- **Beslutningsgrunnlag** (data brukt i check)
- **Timestamp** (ISO 8601 format)

**Bevis** (log inspection):
```bash
# VALID log entries:
2026-02-10 10:34:12 [GATEKEEPER] BLOCKED ZZZUSDT - reason=symbol_not_in_universe, universe_size=567, check_duration_ms=0.3

2026-02-10 10:35:08 [GATEKEEPER] BLOCKED BTCUSDT - reason=max_position_limit_reached, current_positions=10, max_allowed=10, check_duration_ms=0.2

# INVALID log entries:
2026-02-10 10:36:00 [GATEKEEPER] Rejected order  # ❌ No reason
2026-02-10 10:37:00 [GATEKEEPER] BLOCKED  # ❌ No symbol, no reason
```

**Format template**:
```python
logger.info(
    f"[GATEKEEPER] BLOCKED {symbol} - "
    f"reason={reason}, "
    f"rule={rule_name}, "
    f"threshold={threshold}, "
    f"actual={actual_value}, "
    f"check_duration_ms={duration_ms}"
)
```

#### 4. Fail-Mode Definert

**Krav**: **Fail-open** eller **fail-closed** - aldri "silent default"

**Bevis** (code inspection + documentation):
```python
# VALID - Fail-open (explicitly stated):
def check_universe(symbol):
    """
    Check if symbol is in universe allowlist.
    
    Fail mode: FAIL-OPEN
    - If universe unavailable → allow (fallback to default symbols)
    - If Redis timeout → allow
    - If parse error → allow
    
    Rationale: Prevents trading halt on infrastructure failure.
    """
    try:
        universe = redis.get("quantum:cfg:universe:active")
        if not universe:
            logger.warning("Universe unavailable - FAIL-OPEN to default symbols")
            return symbol in DEFAULT_SYMBOLS  # ← Explicit fallback ✅
        
        symbols = json.loads(universe)["symbols"]
        return symbol in symbols
    
    except Exception as e:
        logger.error(f"Gatekeeper exception: {e} - FAIL-OPEN")
        return True  # ← Explicit fail-open behavior ✅

# VALID - Fail-closed (explicitly stated):
def check_risk_limit(order):
    """
    Check if order exceeds risk limits.
    
    Fail mode: FAIL-CLOSED
    - If risk calculator unavailable → reject order
    - If position data missing → reject order
    - Rationale: Unknown risk → no execution
    """
    try:
        current_risk = risk_calculator.get_total_exposure()
        if current_risk is None:
            logger.error("Risk calculator unavailable - FAIL-CLOSED")
            return False  # ← Explicit fail-closed ✅
        
        return current_risk + order.notional < MAX_RISK
    
    except Exception as e:
        logger.error(f"Risk check failed: {e} - FAIL-CLOSED")
        return False  # ← Explicit fail-closed behavior ✅

# INVALID - Ambiguous failure:
def check_something(symbol):
    try:
        result = external_service.check(symbol)
        return result
    except:
        return None  # ❌ What does None mean? Allow or block?
```

**Documentation requirement**:
```markdown
# Component: Universe Gatekeeper

## Fail Mode: FAIL-OPEN

### Rationale
Prevents complete trading halt if universe service crashes.
Default symbols (BTC, ETH, TRX) allow core operations to continue.

### Failure scenarios:
1. Redis timeout → fallback to DEFAULT_SYMBOLS (3 symbols)
2. JSON parse error → fallback to DEFAULT_SYMBOLS
3. Stale data (>5 min) → fallback to DEFAULT_SYMBOLS

### Monitoring:
- Alert if fallback triggered > 3 times/hour
- Dashboard: universe_fallback_count metric
```

#### 5. Scope-Lås

**Krav**: Maks **ÉN dimensjon** per gatekeeper

**Tillatte scopes** (velg én):
- Symbol (allowlist/blocklist)
- Time (market hours, maintenance windows)
- Regime (volatility thresholds, circuit breakers)
- Risk (position limits, notional caps)

**FORBUDT**: Gatekeeper som blokkerer basert på flere dimensjoner:
```python
# INVALID - Multiple dimensions:
class BadGatekeeper:
    def check(self, order):
        if order.symbol not in self.universe:  # ← Dimension 1
            return False
        if order.time not in trading_hours():   # ← Dimension 2 ❌
            return False
        if order.size > position_limit():       # ← Dimension 3 ❌
            return False
        return True

# VALID - Single dimension (separate gatekeepers):
class UniverseGatekeeper:
    def check(self, symbol):
        return symbol in self.allowlist  # ← Only symbol dimension ✅

class TimeGatekeeper:
    def check(self, timestamp):
        return timestamp in trading_hours()  # ← Only time dimension ✅

class RiskGatekeeper:
    def check(self, order):
        return order.size <= position_limit()  # ← Only risk dimension ✅
```

**Critical**: Aldri både entry **OG** exit:
```python
# INVALID - Controls both entry and exit:
class BadGatekeeper:
    def check_entry(self, order): ...   # ← Entry control
    def check_exit(self, order): ...    # ← Exit control ❌

# VALID - Separate components:
class EntryGatekeeper:
    def check(self, entry_order): ...  # ← Entry only ✅

class ExitGatekeeper:
    def check(self, exit_order): ...   # ← Exit only ✅
```

### 🧪 VERIFICATION SCRIPT

```bash
# Template for GATEKEEPER-level verification:

# 1. Hard blocking verification
journalctl -u quantum-<consumer>.service --since "24 hours ago" | \
  grep "BLOCKED\|REJECTED\|executed=False" | wc -l
# Verify: >0 blocked events (gatekeeper is active)

# 2. Determinism verification (100 samples)
for i in {1..100}; do
  redis-cli GET quantum:cfg:universe:active | \
    python3 -c "import json,sys; print('ZZZUSDT' in json.load(sys.stdin)['symbols'])"
done | sort | uniq -c
# Verify: only ONE unique result (deterministic)

# 3. Audit trail verification
journalctl -u quantum-<consumer>.service --since "1 hour ago" | \
  grep "BLOCKED" | head -n 5
# Verify: Each line contains reason, symbol, timestamp

# 4. Fail-mode verification
# Simulate failure:
redis-cli DEL quantum:cfg:universe:active
sleep 5
journalctl -u quantum-<consumer>.service --since "1 min ago" | \
  grep -i "fallback\|fail-open\|fail-closed"
# Verify: Explicit fail-mode message logged

# Restore:
# (restart universe service)

# 5. Scope isolation verification
grep -A 20 "class.*Gatekeeper" /root/quantum_trader/microservices/<gatekeeper>/main.py | \
  grep -E "symbol|time|regime|risk" | wc -l
# Verify: Only ONE dimension mentioned in check logic
```

### 🚫 DISKVALIFISERING

Komponenten faller tilbake til **SCORER** hvis:
- Blokkering kan omgås (soft reject)
- Failure er stille (no logs, no alerts)
- Bruker ML uten deterministisk threshold
- Fail-mode er udefinert eller ambiguous
- Scope er multi-dimensional (controls both entry + exit)
- Audit trail mangler reason eller data

### 🎯 RESULTAT

➡️ **Komponenten kan nekte PnL, men ikke bestemme utfallet**

Eksempler:
- Universe: Blokkerer symbols utenfor allowlist
- Time gatekeeper: Blokkerer trading utenfor market hours
- Risk gatekeeper: Blokkerer orders over position limit

**IKKE eksempler**:
- Harvest Proposal: Beslutter exit (CONTROLLER, ikke gatekeeper)
- AI Ensemble: Genererer intents (OBSERVER, blocked av gatekeeper)

---

## 🟡 → 🟢 GATEKEEPER → CONTROLLER

### ⚠️ HØYESTE KRAV - SJELDEN TILDELING

**Definisjon**: "Denne komponenten får lov til å flytte penger."

### ✅ BEVISKRAV (ALLE, INGEN UNNTAK)

#### 1️⃣ Direkte Execution-Path

**Krav**: Komponenten genererer **ordre eller exit-beslutninger**

**Bevis** (code inspection):
```python
# VALID - Direct order generation:
class ValidController:
    def generate_exit(self, position):
        return {
            "action": "CLOSE",           # ← Direct action ✅
            "symbol": position.symbol,
            "side": "SELL" if position.side == "LONG" else "BUY",
            "qty": position.qty,         # ← Explicit size ✅
            "order_type": "MARKET",
            "reason": "harvest_threshold_reached"
        }

# INVALID - Proxy decisions:
class InvalidController:
    def evaluate(self, position):
        return {
            "should_close": True,  # ← Not an order, just recommendation ❌
            "confidence": 0.8
        }
```

**Critical distinction**:
- **CONTROLLER**: Produces executable orders → `{action, symbol, side, qty}`
- **SCORER**: Produces recommendations → `{score, confidence, reason}`

**Scope requirement**: Velg **ÉN** av:
- Entry (generates OPEN orders)
- Exit (generates CLOSE orders)
- Sizing (modifies qty in existing orders)

**FORBUDT**: Controller som gjør flere:
```python
# INVALID - Multiple roles:
class BadController:
    def decide(self, state):
        entry_order = self.generate_entry(state)   # ← Role 1
        exit_order = self.generate_exit(state)     # ← Role 2 ❌
        size_adjustment = self.adjust_size(state)  # ← Role 3 ❌
        return {...}
```

#### 2️⃣ Counterfactual Proof (OBLIGATORISK)

**Krav**: Minst **1000 trades** hvor PnL med/uten komponenten er målt

**Data sources** (i prioritert rekkefølge):
1. **Production A/B test** (best)
   - 50% trades use controller
   - 50% trades use fallback
   - Same market conditions
   - Minimum 1000 trades per arm

2. **Shadow mode** (acceptable)
   - Controller generates decisions
   - Not executed, but logged
   - Compare hypothetical PnL vs actual PnL
   - Minimum 2000 shadow decisions

3. **Historical replay** (minimum acceptable)
   - Replay past market data
   - Run with/without controller
   - Account for slippage/fees
   - Minimum 5000 historical trades

**Metrics required**:

| Metric | With Controller (A) | Without Controller (B) | Requirement |
|--------|---------------------|------------------------|-------------|
| Total PnL | PnL_A | PnL_B | PnL_A > PnL_B |
| Win rate | WR_A | WR_B | WR_A ≥ WR_B - 5% |
| Max drawdown | DD_A | DD_B | DD_A < DD_B * 0.9 |
| Sharpe ratio | SR_A | SR_B | SR_A > SR_B |
| P-value | - | - | p < 0.05 |

**Statistical significance**:
```python
from scipy.stats import ttest_ind

def validate_controller(trades_with, trades_without):
    """
    Validate controller improves PnL with statistical significance.
    
    Args:
        trades_with: List of PnL values with controller
        trades_without: List of PnL values without controller
    
    Returns:
        dict with validation results
    """
    # 1. Sample size check
    assert len(trades_with) >= 1000, "Need ≥1000 trades with controller"
    assert len(trades_without) >= 1000, "Need ≥1000 trades without controller"
    
    # 2. Mean PnL improvement
    mean_with = np.mean(trades_with)
    mean_without = np.mean(trades_without)
    improvement_pct = (mean_with - mean_without) / abs(mean_without) * 100
    
    # 3. Statistical test
    t_stat, p_value = ttest_ind(trades_with, trades_without)
    
    # 4. Drawdown analysis
    dd_with = calculate_max_drawdown(trades_with)
    dd_without = calculate_max_drawdown(trades_without)
    dd_improvement_pct = (dd_without - dd_with) / dd_without * 100
    
    return {
        "mean_pnl_improvement_pct": improvement_pct,
        "p_value": p_value,
        "significant": p_value < 0.05 and improvement_pct > 0,
        "drawdown_improvement_pct": dd_improvement_pct,
        "sample_size_with": len(trades_with),
        "sample_size_without": len(trades_without)
    }
```

**Example report**:
```markdown
# Controller Validation: Harvest Proposal System

## Test Period: Jan 15 - Feb 10, 2026

### Method: Production A/B Test
- Group A: Harvest exits (1247 trades)
- Group B: Time-based exits (1198 trades)

### Results:

| Metric | Harvest (A) | Time-based (B) | Improvement |
|--------|-------------|----------------|-------------|
| Mean PnL/trade | $2.34 | $1.87 | +25% ✅ |
| Win rate | 58% | 54% | +4% ✅ |
| Max drawdown | $124 | $187 | -34% ✅ |
| Sharpe ratio | 1.83 | 1.42 | +29% ✅ |
| P-value | - | - | 0.003 ✅ |

### Conclusion: ✅ APPROVED for CONTROLLER level
Harvest Proposal demonstrates statistically significant improvement
in realized PnL with acceptable drawdown reduction.
```

**Diskvalifisering**: Hvis test viser:
- p-value ≥ 0.05 (not significant)
- PnL_A ≤ PnL_B (no improvement)
- DD_A ≥ DD_B (higher drawdown)
- Sample size < 1000 per arm

#### 3️⃣ Failure Safety

**Krav**: Hvis controller krasjer → trading fortsetter trygt

**Bevis** (failure mode test):
```bash
# Test script:

# 1. Verify baseline (controller active)
systemctl status quantum-<controller>.service
# Verify: active (running)

journalctl -u quantum-intent-executor.service --since "1 hour ago" | \
  grep "executed=True" | wc -l
# Record: baseline_execution_count

# 2. Simulate controller crash
systemctl stop quantum-<controller>.service
sleep 60

# 3. Verify fallback behavior
journalctl -u quantum-intent-executor.service --since "1 min ago" | \
  grep -i "fallback\|controller.*unavailable"
# Verify: Explicit fallback message logged ✅

# 4. Verify safe operation
journalctl -u quantum-intent-executor.service --since "1 min ago" | \
  grep "executed=True" | wc -l
# Verify: execution_count > 0 (trading continues) ✅

# 5. Verify no silent behavior change
journalctl -u quantum-intent-executor.service --since "1 min ago" | \
  grep -E "ERROR|CRITICAL|Traceback"
# Verify: No exceptions or errors ✅

# 6. Restore and verify
systemctl start quantum-<controller>.service
sleep 30
systemctl status quantum-<controller>.service
# Verify: active (running), auto-recovery ✅
```

**Fallback requirement**:
```python
# VALID - Explicit fallback:
def get_exit_decision(position):
    try:
        controller = get_controller()
        if controller and controller.is_healthy():
            return controller.decide_exit(position)
        else:
            logger.warning("Controller unavailable - using fallback exit logic")
            return fallback_exit_logic(position)  # ← Explicit fallback ✅
    except Exception as e:
        logger.error(f"Controller failed: {e} - using fallback")
        return fallback_exit_logic(position)  # ← Exception safety ✅

def fallback_exit_logic(position):
    """
    Conservative exit logic used when controller unavailable.
    
    Rules:
    - Max position duration: 24 hours
    - Emergency stop-loss: -5% from entry
    - No leverage adjustment
    """
    return {
        "action": "CLOSE",
        "reason": "fallback_timeout_or_stoploss",
        ...
    }
```

**Diskvalifisering**: Hvis controller crash causes:
- Trading halt (no fallback)
- Silent behavior change (no logs)
- Exception propagation (consumer crashes)
- Data loss (positions orphaned)

#### 4️⃣ Scope Singularity

**Krav**: Controller has **ÉN** og **KUN ÉN** rolle

**Tillatte scopes** (velg én):
- **Entry controller**: Generates OPEN orders (not exits)
- **Exit controller**: Generates CLOSE orders (not entries)
- **Sizing controller**: Modifies position size (not entry/exit timing)

**Strenge separasjonsregler**:

| Controller Type | Allowed | Forbidden |
|----------------|---------|-----------|
| Entry | Generate OPEN orders | Generate CLOSE orders ❌ |
| Exit | Generate CLOSE orders | Generate OPEN orders ❌ |
| Sizing | Modify qty field | Decide entry/exit timing ❌ |

**Code inspection**:
```python
# VALID - Exit controller (Harvest Proposal):
class HarvestController:
    def decide_exit(self, position):
        """
        Exit-only controller.
        Scope: CLOSE orders based on R-value thresholds.
        """
        if position.r_net >= self.target_r:
            return {"action": "CLOSE", ...}  # ← Exit only ✅
        return None
    
    # NO entry methods ✅

# INVALID - Multi-role controller:
class BadController:
    def decide_entry(self, signal):      # ← Role 1: Entry
        return {"action": "OPEN", ...}
    
    def decide_exit(self, position):     # ← Role 2: Exit ❌
        return {"action": "CLOSE", ...}
    
    def adjust_size(self, order):        # ← Role 3: Sizing ❌
        order.qty *= 1.5
        return order
```

**Verification**:
```bash
# Check controller scope:
grep -E "def.*(entry|exit|size|open|close)" \
  /root/quantum_trader/microservices/<controller>/main.py

# Count roles:
# - If 1 role found → ✅ Valid single scope
# - If 2+ roles found → ❌ Invalid multi-scope
```

#### 5️⃣ Kill Switch

**Krav**: Umiddelbar deaktivering mulig med **verifiserbar effekt**

**Implementation requirement**:
```bash
# Kill switch must be ONE command:

# Option A: Systemd service
systemctl stop quantum-<controller>.service

# Option B: Feature flag
redis-cli SET quantum:cfg:<controller>:enabled false

# Option C: Environment variable + restart
export CONTROLLER_ENABLED=false
systemctl restart quantum-intent-executor.service
```

**Verification requirement** (within 60 seconds):
```bash
# 1. Trigger kill switch
systemctl stop quantum-harvest.service

# 2. Wait maximum 60 seconds
sleep 60

# 3. Verify no new controller decisions
journalctl -u quantum-intent-executor.service --since "1 min ago" | \
  grep "harvest.intent\|harvest_executed"
# Verify: Zero new harvest events ✅

# 4. Verify fallback active
journalctl -u quantum-intent-executor.service --since "1 min ago" | \
  grep "fallback\|controller.*disabled"
# Verify: Fallback message present ✅

# 5. Verify no orphaned positions
# (manual check or automated position reconciliation)
```

**Alert requirement**:
```python
# Kill switch must trigger monitoring alert:

if not controller_enabled():
    alert_manager.send(
        severity="WARNING",
        component="harvest_controller",
        message="Controller disabled via kill switch",
        action_required="Verify fallback logic active"
    )
```

**Diskvalifisering**: Hvis kill switch:
- Requires multiple steps
- Takes > 60 seconds to take effect
- Leaves orphaned positions
- Fails silently (no verification possible)
- Requires code deploy to disable

### 🧪 VERIFICATION SCRIPT (CONTROLLER)

```bash
# Complete CONTROLLER-level verification:

# 1. Execution-path verification
redis-cli XREVRANGE quantum:stream:<controller_output> + - COUNT 5 | \
  grep -E "action.*CLOSE|action.*OPEN|qty"
# Verify: Direct order fields present ✅

# 2. Counterfactual proof verification
# (requires separate analysis script, see counterfactual_report.py)
python3 scripts/validate_controller_counterfactual.py \
  --controller=harvest \
  --test-period="2026-01-15:2026-02-10" \
  --min-trades=1000
# Verify: p-value < 0.05, improvement > 0% ✅

# 3. Failure safety test
systemctl stop quantum-<controller>.service
sleep 60
journalctl -u quantum-intent-executor.service --since "1 min ago" | \
  grep -i "fallback\|error"
# Verify: Explicit fallback, no errors ✅
systemctl start quantum-<controller>.service

# 4. Scope singularity verification
grep -c "def.*entry\|def.*open" \
  /root/quantum_trader/microservices/<controller>/main.py
grep -c "def.*exit\|def.*close" \
  /root/quantum_trader/microservices/<controller>/main.py
# Verify: Only ONE type is > 0 (not both) ✅

# 5. Kill switch test
systemctl stop quantum-<controller>.service
sleep 60
journalctl -u quantum-intent-executor.service --since "1 min ago" | \
  grep "<controller_name>"
# Verify: Zero new events from controller ✅
systemctl start quantum-<controller>.service
```

### 🚫 DISKVALIFISERING

Komponenten faller tilbake til **GATEKEEPER** hvis:
- ML-modell uten forklarbarhet (black box decisions)
- Flere roller (entry + exit, eller exit + sizing)
- Manglende counterfactual proof (p ≥ 0.05 eller sample < 1000)
- Ingen fallback på failure (trading halts)
- Kill switch krever > 60 sekunder
- PnL improvement ikke statistisk signifikant

### 🎯 RESULTAT

➡️ **Komponenten får begrenset, eksplisitt pengemakt**

**Current example**:
- ✅ Harvest Proposal: Exit controller (R-value based CLOSE orders)

**Future candidates** (must prove themselves):
- ❌ AI Entry Signal: Needs counterfactual proof
- ❌ Adaptive Leverage: Needs scope singularity (sizing only)
- ❌ RL Agent: Needs statistical validation

---

## 🧠 META-REGEL (DEN VIKTIGSTE)

### Autoritet gis IKKE fordi noe er smart

Autoritet gis fordi:
1. **Systemet blir målt bedre** (counterfactual proof)
2. **Failure er akseptabelt** (safe fallback)
3. **Scope er begrenset** (single responsibility)

### Hvis du ikke kan bevise det → forblir komponenten på sitt nivå

**Ingen unntak.**

---

## 📌 PRAKTISK BRUK (ANBEFALT WORKFLOW)

For hver komponent som skal eskaleres:

### 1. Current State Assessment
```markdown
# Component: AI Exit Evaluator

Current Level: OBSERVER
Requested Level: SCORER

Evidence checklist:
- [x] Service running (OBSERVER base requirement)
- [x] Continuous output (39k events produced)
- [ ] Output consumed (❌ No consumers found)
- [ ] Decision contact (❌ Not used in any branches)
- [ ] Non-authoritarian (❌ No fallback needed, ignored)
- [ ] Reproducible effect (⏸️ Not testable without consumer)
- [ ] Isolated impact (⏸️ Not applicable without consumer)

Result: ❌ Missing 3/5 SCORER requirements
Action: Cannot escalate until consumer path established
```

### 2. Gap Analysis
```markdown
# Missing Evidence for OBSERVER → SCORER

1. Consumer Path (CRITICAL):
   - No component reads ai.exit.decision stream
   - Need: intent_executor or harvest to consume output
   - Effort: Medium (integration work)

2. Decision Contact (CRITICAL):
   - Output not used in any if/threshold/ranking
   - Need: Use score in harvest timing or exit ranking
   - Effort: Low (code change)

3. Fallback Behavior (REQUIRED):
   - Not applicable until consumer exists
   - Need: Define fallback if exit evaluator fails
   - Effort: Low (documentation + code)
```

### 3. Implementation Roadmap
```markdown
# Roadmap: AI Exit Evaluator → SCORER

Phase 1: Establish Consumer (Week 1)
- [ ] Modify harvest proposal to read ai.exit.decision
- [ ] Add decision contact point (score > 0.7 → expedite exit)
- [ ] Define fallback (use R-value only if evaluator fails)
- [ ] Verify determinism (same state → same score)

Phase 2: Verification (Week 2)
- [ ] Run verification script (consumer check)
- [ ] Test fallback behavior (stop evaluator, verify harvest continues)
- [ ] Document isolated impact (exit timing only, not entry)

Phase 3: Monitoring (Week 3)
- [ ] Deploy to production
- [ ] Monitor lag < 100 (consumer keeping up)
- [ ] Track fallback trigger rate (should be <1%)
- [ ] Collect counterfactual data for future GATEKEEPER consideration

Phase 4: Approval (Week 4)
- [ ] Update PNL_AUTHORITY_MAP_CANONICAL.md
- [ ] Change status: OBSERVER → SCORER
- [ ] Document evidence in decision log
```

### 4. NO DISCUSSION - ONLY CHECKLIST

**Before requesting escalation**:
```markdown
# Escalation Request Template

Component: <name>
From: <current_level>
To: <requested_level>

Evidence (ALL must be checked):
- [ ] <requirement 1>
- [ ] <requirement 2>
- [ ] <requirement 3>
...

Verification:
- [ ] Ran verification script (output attached)
- [ ] Logs show expected behavior (sample attached)
- [ ] Tests pass (test report attached)

Counterfactual (if applicable):
- [ ] ≥1000 trades analyzed
- [ ] p-value < 0.05
- [ ] Improvement > 0%
- [ ] Report attached

Approval: _______________  Date: _______________
```

**If ANY checkbox is unchecked → REJECTED immediately**

---

## 🔐 GOVERNANCE

### Review Process

1. **Self-assessment**: Component owner fills escalation checklist
2. **Peer review**: Another engineer verifies evidence
3. **Automated check**: CI runs verification scripts
4. **Approval**: Only if ALL checks pass

### Demotion Triggers

Component is **automatically demoted** if:
- Evidence disappears (e.g., consumer stops reading)
- Failure mode violated (e.g., crashes without fallback)
- Counterfactual invalidated (e.g., performance degrades)

### Audit Frequency

- **CONTROLLER** level: Monthly audit (verify counterfactual still valid)
- **GATEKEEPER** level: Quarterly audit (verify fail-mode correct)
- **SCORER** level: Bi-annual audit (verify still consumed)
- **OBSERVER** level: No audit (passive monitoring)

---

## 📊 REFERENCE: QUICK DECISION MATRIX

| Question | Answer = | Level |
|----------|----------|-------|
| Service running + output growing? | No | DEAD |
| Service running + output growing? | Yes | OBSERVER |
| Output consumed by decision logic? | Yes | SCORER |
| Can block execution? | Yes | GATEKEEPER |
| Generates executable orders? | Yes | CONTROLLER (if counterfactual proven) |

---

## ✅ VERSION HISTORY

**V1.0** (Feb 10, 2026):
- Initial rulebook
- Based on 3 runtime audits (CLM, Prediction Agents, Governance/Universe)
- Canonical reference for all future escalations

**Next review**: March 10, 2026 or upon first escalation request

---

**END OF RULEBOOK**
