# 🛑 BASELINE SAFETY CONTROLLER PROMPT — CANONICAL

**Document Type:** Emergency Controller Specification  
**Authority Level:** 🟢 CONTROLLER (RESTRICTED)  
**Intended Lifetime:** Midlertidig (kun under Authority Freeze)  
**Design Principle:** Capital Preservation Only  
**Version:** 1.0  
**Date:** February 10, 2026  

---

## 🎯 FORMÅL

**Baseline Safety Controller (BSC)** eksisterer kun for å forhindre ukontrollert tap når:

1. Ingen fullverdig CONTROLLER er godkjent
2. Authority Freeze er aktiv
3. Systemet ellers ville vært helt uten exit-beskyttelse

---

### ⚠️ KRITISK FORSTÅELSE:

```
BSC er IKKE en trading-strategi
BSC er IKKE en AI
BSC er IKKE optimal
BSC er IKKE ønskelig long-term

BSC er en LIFEBOAT
```

**Den gjør én ting:**  
Sørger for at tap ikke løper ukontrollert mens systemet er i Authority Freeze.

---

## 🚨 AKTIVERINGSKRITERIER

### ✅ BSC KAN AKTIVERES hvis og bare hvis:

| Kriterie | Status | Verifikasjon |
|----------|--------|--------------|
| 1. Authority Freeze er aktiv | REQUIRED | Check: AUTHORITY_FREEZE_PROMPT_CANONICAL.md |
| 2. Det finnes åpne posisjoner | REQUIRED | Check: Redis `quantum:position:*` keys |
| 3. Ingen annen 🟢 CONTROLLER autorisert | REQUIRED | Check: PNL_AUTHORITY_MAP_CANONICAL.md |
| 4. Aktivering er eksplisitt dokumentert | REQUIRED | Event: BASELINE_SAFETY_ACTIVATED |

**All 4 må være oppfylt.**

---

### ❌ BSC SKAL IKKE AKTIVERES hvis:

| Scenario | Reason | Alternative |
|----------|--------|-------------|
| 0 åpne posisjoner | Ingen risiko å beskytte mot | Wait for full CONTROLLER |
| Godkjent full CONTROLLER eksisterer | BSC redundant | Use full CONTROLLER |
| Authority Freeze ikke aktiv | System operates normally | No emergency fallback needed |

---

## 🔐 AUTORITET (STERKT BEGRENSET)

Baseline Safety Controller har **kun én tillatt handling:**

---

### ✅ TILLATT (1 handling)

| Action | Scope | Trigger |
|--------|-------|---------|
| **FORCE CLOSE** eksisterende posisjoner | Exit only | Hard safety breach detected |

**Eksempel:**
```python
# ONLY allowed operation:
if safety_breach_detected(position):
    binance.create_market_order(
        symbol=position.symbol,
        side='SELL' if position.side == 'LONG' else 'BUY',
        amount=abs(position.quantity)
    )
    log_event("BASELINE_SAFETY_CLOSE", reason=breach_type)
```

---

### ❌ FORBUDT (ABSOLUTT — 8 categories)

| Action | Status | Rationale |
|--------|--------|-----------|
| ❌ Åpne nye posisjoner | BLOCKED | BSC = exit-only, never entry |
| ❌ Endre sizing / leverage | BLOCKED | No position modification outside close |
| ❌ Optimalisere exits | BLOCKED | Not a trading strategy |
| ❌ Delvise exits | BLOCKED | Full close only (simplicity) |
| ❌ Trailing logic | BLOCKED | No dynamic optimization |
| ❌ ML / heuristikk / scoring | BLOCKED | Fixed thresholds only |
| ❌ Tidsbaserte exits uten tap | BLOCKED | Only safety-triggered closes |
| ❌ Regime-avhengig logikk | BLOCKED | Global rules, no adaptivity |

**Principle:**  
> "If it's smarter than an if-statement, it doesn't belong in BSC."

---

## 🧱 BESLUTNINGSLOGIKK (KANONISK)

BSC opererer med **faste, globale grenser** (ingen ML, ingen dynamikk):

---

### 📐 CANONICAL ALGORITHM:

```python
# PSEUDO-CODE (normative specification)

FOR EACH open_position:
    
    # TRIGGER 1: Max loss breach
    IF position.unrealized_pnl_pct <= -MAX_LOSS_PCT:
        FORCE_CLOSE(position, reason="MAX_LOSS_BREACH")
        CONTINUE
    
    # TRIGGER 2: Max duration exceeded
    IF position.duration_hours >= MAX_DURATION_HOURS:
        FORCE_CLOSE(position, reason="MAX_DURATION_BREACH")
        CONTINUE
    
    # TRIGGER 3: Liquidation risk
    IF position.margin_ratio >= LIQUIDATION_THRESHOLD:
        FORCE_CLOSE(position, reason="LIQUIDATION_RISK")
        CONTINUE

# No other logic allowed
```

---

### 🎛️ DEFAULT PARAMETERS:

| Parameter | Default Value | Rationale |
|-----------|---------------|-----------|
| `MAX_LOSS_PCT` | **-3.0%** | Conservative stop-loss (prevents catastrophic loss) |
| `MAX_DURATION_HOURS` | **72h** | Force exit stale positions (3 days max) |
| `LIQUIDATION_THRESHOLD` | **0.85** | Close at 85% margin ratio (before exchange liquidates) |

**These are FIXED.** No regime adaptation, no ML tuning, no dynamic adjustment.

---

### ⚙️ IMPLEMENTATION REQUIREMENTS:

1. **Check frequency:** Every 60 seconds minimum
2. **Execution:** Direct Binance API (bypass intent pipeline)
3. **Order type:** MARKET (guaranteed fill, no optimization)
4. **Retry logic:** 3 attempts, then MANUAL_ALERT
5. **Logging:** Every close + every check (audit trail)

---

## 📍 SCOPE-LÅS (IMMUTABLE)

| Dimension | BSC Authority | Full CONTROLLER Authority |
|-----------|---------------|---------------------------|
| **Entry** | ❌ BLOCKED | ✅ Allowed |
| **Exit** | ✅ (kun nød) | ✅ (full control) |
| **Sizing** | ❌ BLOCKED | ✅ Allowed |
| **Symbolvalg** | ❌ BLOCKED | ✅ Allowed |
| **Timing-optimalisering** | ❌ BLOCKED | ✅ Allowed |
| **Regime-awareness** | ❌ BLOCKED | ✅ Allowed |
| **ML/AI** | ❌ BLOCKED | ✅ Allowed |

**BSC kan kun lukke. Aldri åpne. Aldri optimalisere.**

---

## 🧠 FAIL-MODE (IKKE FORHANDLBAR)

### ⚠️ HVIS BSC KRASJER:

```
BSC FAIL MODE = FAIL OPEN
```

**Meaning:**

| Event | BSC Behavior | System Behavior |
|-------|--------------|-----------------|
| BSC service crashes | Log error, stop execution | Positions remain open |
| BSC logic error | Skip position, log error | Position unaffected |
| Binance API failure | Retry 3x, then alert | Manual intervention required |
| Redis connection loss | Log error, skip check | No forced closes |

**Rationale:**  
> Better to require manual intervention than to execute bad/partial closes.

---

### ❌ BSC SKAL ALDRI:

- ❌ Eskalere egen autoritet automatisk
- ❌ Bypasse safety checks "in emergency"
- ❌ Retry indefinitely (max 3 attempts)
- ❌ Modify thresholds dynamically
- ❌ Switch to "smart mode" on failure

**BSC stays dumb, or BSC stops.**

---

## 🧾 OBLIGATORISK LOGGING

### 📝 EVERY ACTION MUST LOG:

#### 1. **Close Event** (when BSC closes position):

```json
{
  "event": "BASELINE_SAFETY_CLOSE",
  "timestamp": "2026-02-10T20:15:32Z",
  "symbol": "BTCUSDT",
  "side": "LONG",
  "quantity": 0.05,
  "reason": "MAX_LOSS_BREACH",
  "trigger_values": {
    "unrealized_pnl_pct": -3.4,
    "duration_hours": 48,
    "margin_ratio": 0.65
  },
  "authority_mode": "AUTHORITY_FREEZE",
  "controller": "BASELINE_SAFETY_CONTROLLER",
  "order_id": "1234567890",
  "execution_price": 42350.50,
  "realized_pnl": -145.23
}
```

#### 2. **Health Check** (every 60s):

```json
{
  "event": "BASELINE_SAFETY_CHECK",
  "timestamp": "2026-02-10T20:14:00Z",
  "positions_checked": 1,
  "breaches_detected": 0,
  "status": "ACTIVE"
}
```

#### 3. **Activation Event** (on deployment):

```json
{
  "event": "BASELINE_SAFETY_ACTIVATED",
  "timestamp": "2026-02-10T19:30:00Z",
  "reason": "AUTHORITY_FREEZE_WITH_OPEN_POSITIONS",
  "parameters": {
    "max_loss_pct": -3.0,
    "max_duration_hours": 72,
    "liquidation_threshold": 0.85
  },
  "documentation": "BASELINE_SAFETY_CONTROLLER_PROMPT_CANONICAL.md"
}
```

#### 4. **Deactivation Event** (when replaced):

```json
{
  "event": "BASELINE_SAFETY_DEACTIVATED",
  "timestamp": "2026-02-14T10:00:00Z",
  "reason": "FULL_CONTROLLER_APPROVED",
  "replacement": "Harvest Proposal (re-escalated)",
  "audit_reference": "HARVEST_PROPOSAL_ESCALATION_AUDIT_FEB14_2026.md"
}
```

**All logs must be queryable for audit (journalctl + Redis).**

---

## 🧊 SAMSPILL MED AUTHORITY FREEZE

### 🔒 BSC UNDER FREEZE:

| BSC Behavior | Effect on Freeze | Authority Map Status |
|--------------|------------------|----------------------|
| BSC operates | ✅ Freeze remains active | BSC listed as CONTROLLER (restricted) |
| BSC closes position | ✅ Freeze remains active | No change to other components |
| BSC deactivated | ⚠️ Freeze status unchanged | BSC removed from authority map |

**Critical:**
```
BSC opprettholder Authority Freeze
BSC opphever IKKE freeze
BSC blokkerer IKKE fremtidig eskalering
```

---

### 🔄 INTERAKSJONER:

**1. BSC + Full CONTROLLER candidate:**
- BSC continues operating during audit
- If audit passes → BSC deactivated, replaced
- If audit fails → BSC continues

**2. BSC + Repair attempt (PATH 1):**
- BSC operates during Harvest Brain repair
- Provides safety net during repair work
- Deactivated when Harvest verified working

**3. BSC + Other OBSERVER components:**
- All OBSERVER components continue telemetry
- BSC does not interfere with observation
- BSC only acts on safety triggers

---

## 🔁 DEAKTIVERING

### ✅ BSC SKAL DEAKTIVERES når:

| Trigger | Required Evidence | Process |
|---------|-------------------|---------|
| 1. Ny 🟢 CONTROLLER godkjent | Full escalation audit passed | Log DEACTIVATED event |
| 2. Authority Freeze opphevet | AUTHORITY_UNFREEZE logged | Remove BSC from authority map |
| 3. Alle posisjoner lukket | 0 open positions | BSC no longer needed |
| 4. Manuell beslutning | Explicit deactivation command | Document reason |

**Process:**
```bash
# 1. Stop BSC service
systemctl stop quantum-baseline-safety.service

# 2. Log deactivation
redis-cli XADD quantum:stream:authority.events * \
  event BASELINE_SAFETY_DEACTIVATED \
  timestamp $(date -u +%Y-%m-%dT%H:%M:%SZ) \
  reason FULL_CONTROLLER_APPROVED

# 3. Update authority map
# (via formal process, not automated)

# 4. Verify freeze status
# (may remain active if no replacement)
```

---

### ❌ BSC SKAL IKKE DEAKTIVERES hvis:

- Authority Freeze aktiv + open positions + no replacement CONTROLLER
- Repair attempt in progress (unless explicitly verified)
- "Testing" or "temporary" scenarios

**BSC deaktiveres kun med eksplisitt erstatter ELLER når den ikke lenger er nødvendig.**

---

## 🎯 DEPLOYMENT CHECKLIST

### ✅ PRE-DEPLOYMENT VERIFICATION:

- [ ] Authority Freeze confirmed active
- [ ] Open positions confirmed (> 0)
- [ ] No other CONTROLLER authorized
- [ ] Parameters configured (MAX_LOSS_PCT, MAX_DURATION_HOURS, etc.)
- [ ] Direct Binance API access verified
- [ ] Logging pipeline verified (journalctl + Redis)
- [ ] Fail-open behavior tested
- [ ] Manual override procedure documented

### ✅ POST-DEPLOYMENT VERIFICATION:

- [ ] BASELINE_SAFETY_ACTIVATED event logged
- [ ] Health checks running (60s interval)
- [ ] Position monitoring active
- [ ] Authority map updated (BSC listed as CONTROLLER)
- [ ] No unintended closes in first 10 minutes
- [ ] Logs accessible for audit

### ✅ ONGOING MONITORING:

- [ ] Daily: Verify BSC health checks continue
- [ ] Daily: Check for any close events (investigate if unexpected)
- [ ] Weekly: Review if BSC still needed (push for full CONTROLLER)
- [ ] Monthly: Audit BSC behavior (no scope creep)

---

## 🧠 META-PRINSIPP (KANONISK)

> **"Dette er ikke intelligens. Det er ansvar."**

### KOROLLARER:

1. **BSC er en failure admission**
   - System admits: "Vi har ingen god CONTROLLER nå"
   - Not a permanent solution

2. **BSC er midlertidig by design**
   - Intended lifetime: Days to weeks, NOT months
   - Push for full CONTROLLER escalation

3. **BSC er dum by design**
   - No ML, no optimization, no adaptivity
   - Simplicity = auditability = trust

4. **BSC beskytter, ikke performer**
   - Goal: Prevent catastrophic loss
   - NOT goal: Maximize returns

5. **BSC er siste utvei**
   - Better than nothing
   - Worse than proper CONTROLLER

---

## 📚 IMPLEMENTATION REFERENCE

### 🐍 PYTHON SKELETON:

```python
# baseline_safety_controller.py (reference implementation)

import time
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class BSCConfig:
    MAX_LOSS_PCT: float = -3.0
    MAX_DURATION_HOURS: int = 72
    LIQUIDATION_THRESHOLD: float = 0.85
    CHECK_INTERVAL_SECONDS: int = 60

class BaselineSafetyController:
    """Emergency exit-only controller for Authority Freeze mode."""
    
    def __init__(self, config: BSCConfig):
        self.config = config
        self.active = False
        
    def activate(self):
        """Activate BSC (logs activation event)."""
        log_event("BASELINE_SAFETY_ACTIVATED", 
                  parameters=self.config.__dict__)
        self.active = True
        
    def deactivate(self, reason: str):
        """Deactivate BSC (logs deactivation event)."""
        log_event("BASELINE_SAFETY_DEACTIVATED", reason=reason)
        self.active = False
        
    def check_positions(self, positions: List[Position]):
        """Check all positions for safety breaches."""
        log_event("BASELINE_SAFETY_CHECK", 
                  positions_checked=len(positions))
        
        for pos in positions:
            breach = self._detect_breach(pos)
            if breach:
                self._force_close(pos, breach.reason)
                
    def _detect_breach(self, pos: Position) -> Optional[Breach]:
        """Pure function: detect if position breaches safety rules."""
        
        # TRIGGER 1: Max loss
        if pos.unrealized_pnl_pct <= self.config.MAX_LOSS_PCT:
            return Breach(reason="MAX_LOSS_BREACH", 
                         value=pos.unrealized_pnl_pct)
        
        # TRIGGER 2: Max duration
        if pos.duration_hours >= self.config.MAX_DURATION_HOURS:
            return Breach(reason="MAX_DURATION_BREACH", 
                         value=pos.duration_hours)
        
        # TRIGGER 3: Liquidation risk
        if pos.margin_ratio >= self.config.LIQUIDATION_THRESHOLD:
            return Breach(reason="LIQUIDATION_RISK", 
                         value=pos.margin_ratio)
        
        return None
        
    def _force_close(self, pos: Position, reason: str):
        """Execute emergency close (market order, no optimization)."""
        
        # Direct Binance API call (bypass intent pipeline)
        order = binance.create_market_order(
            symbol=pos.symbol,
            side='SELL' if pos.side == 'LONG' else 'BUY',
            amount=abs(pos.quantity)
        )
        
        # Log every close (audit trail)
        log_event("BASELINE_SAFETY_CLOSE",
                  symbol=pos.symbol,
                  reason=reason,
                  order_id=order.id,
                  realized_pnl=order.realized_pnl)
        
    def run(self):
        """Main loop (runs until deactivated)."""
        while self.active:
            try:
                positions = get_open_positions()
                self.check_positions(positions)
                time.sleep(self.config.CHECK_INTERVAL_SECONDS)
            except Exception as e:
                log_error("BSC_CHECK_FAILED", error=str(e))
                # FAIL OPEN: continue loop, skip this iteration

# Usage:
# bsc = BaselineSafetyController(BSCConfig())
# bsc.activate()
# bsc.run()  # Blocks until deactivated
```

---

## ✅ KANONISK STATUS

### DETTE DOKUMENT DEFINERER:

1. **Hva BSC er** (emergency exit-only controller)
2. **Når BSC aktiveres** (freeze + positions + no controller)
3. **Hva BSC kan gjøre** (force close ONLY)
4. **Hva BSC IKKE kan gjøre** (8 forbidden categories)
5. **Hvordan BSC fungerer** (3 fixed triggers, no ML)
6. **Hvordan BSC feiler** (FAIL OPEN, never escalate)
7. **Hvordan BSC deaktiveres** (replaced by full CONTROLLER)

---

## 🔗 RELATED DOCUMENTS

**Authority Governance Framework:**

1. **PNL_AUTHORITY_ESCALATION_RULEBOOK_V1.md**  
   How BSC would be escalated (if not emergency exception)

2. **PNL_AUTHORITY_DEMOTION_PROMPT_CANONICAL.md**  
   How BSC would be demoted (if breaches scope)

3. **AUTHORITY_FREEZE_PROMPT_CANONICAL.md**  
   Context in which BSC operates

4. **BASELINE_SAFETY_CONTROLLER_AUDIT_PROMPT_CANONICAL.md**  
   Comprehensive weekly audit (boundary enforcement)

5. **BSC_SCOPE_GUARD_DAILY_AUDIT.md**  
   Daily operational scope verification (simple pass/fail)

6. **PNL_AUTHORITY_MAP_CANONICAL_FEB10_2026.md**  
   Where BSC would be listed (as CONTROLLER with restrictions)

7. **NO_CONTROLLER_MODE_DECLARATION_FEB10_2026.md**  
   PATH 2 (this is the implementation of that path)

---

## 📌 SLUTTSETNING (KANONISK)

> **"Når systemet ikke vet hva som er riktig,  
> vet det i det minste hva som er farlig."**

### FINAL PRINCIPLE:

```
BSC exists at the intersection of three truths:

1. Authority must be earned (escalation required)
2. Capital must be protected (can't wait for perfect)
3. Simplicity enables trust (dumb beats clever)

BSC is the minimum viable controller.
Nothing more.
Nothing less.
```

---

**End of Baseline Safety Controller Specification**

**Signed:** PnL Authority Framework  
**Version:** 1.0 (Canonical)  
**Date:** 2026-02-10 19:45 UTC  
**Status:** SPECIFICATION (not yet deployed)  
**Implementation:** Pending user decision (PATH 2 from NO_CONTROLLER_MODE)
