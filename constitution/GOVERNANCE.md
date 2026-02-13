# 🏛️ GOVERNANCE - Decision Hierarchy & Constitutional Laws

**Document**: Constitutional Governance Framework  
**Authority**: SUPREME  
**Version**: 1.0  

---

## The 15 Grunnlover (Constitutional Laws)

These laws are **immutable** during trading operations.  
No code, no human, no circumstance can override them.

---

### §1 Kontrollert Størrelse Per Trade

> **"Risiko per trade skal aldri overstige policy-definert maksimum"**

| Parameter | Value |
|-----------|-------|
| Max Risk | 2% of equity |
| Enforcement | AUTOMATIC |
| Override | ❌ FORBIDDEN |

**Implementation**: `services/risk_kernel/position_limits.py`

---

### §2 Maks Daglig Tap Aktiverer Full Stopp

> **"Ved maks daglig tap stoppes ALL trading umiddelbart"**

| Parameter | Value |
|-----------|-------|
| Trigger | 5% daily loss |
| Effect | FULL HALT |
| Resume | Next trading day + review |

**Implementation**: `services/risk_kernel/daily_limits.py`

---

### §3 Aldri Øk Taper-Posisjon

> **"Det er forbudt å legge til en posisjon som er i tap"**

| Scenario | Allowed |
|----------|---------|
| Add to winner | ✅ Yes (with limits) |
| Add to loser | ❌ NEVER |
| Average down | ❌ NEVER |

**Implementation**: `services/entry_gate/entry_blocker.py`

---

### §4 Likvider Ved Kritisk Margin

> **"Posisjoner likvideres automatisk ved kritisk marginnivå"**

| Level | Action |
|-------|--------|
| < 200% maintenance | Warning |
| < 150% maintenance | Reduce 50% |
| < 120% maintenance | Emergency close ALL |

**Implementation**: `services/risk_kernel/margin_safety.py`

---

### §5 Ignorer AI Ved Basisbrudd

> **"AI-signaler ignoreres fullstendig ved brudd på grunnleggende regler"**

AI signals are **advisory only**. They are rejected when:
- Any Grunnlov would be violated
- Risk limits exceeded
- Data integrity compromised
- System health degraded

**Implementation**: `services/policy_engine/enforcement.py`

---

### §6 Tving Exit Ved Data-Gap

> **"Manglende eller korrupt data trigger umiddelbar exit"**

| Gap Type | Response |
|----------|----------|
| Price data missing | Close position |
| Volume data corrupt | Halt new entries |
| API disconnect > 30s | Emergency flat |

**Implementation**: `services/data_integrity/gap_detector.py`

---

### §7 Flat Ved Ekstrem Funding

> **"Ekstrem funding rate trigger flat posisjon"**

| Funding Rate | Action |
|--------------|--------|
| Normal range | Continue |
| P95 (warning) | Reduce size |
| P99 (extreme) | Close ALL |

**Implementation**: `services/market_regime/funding_monitor.py`

---

### §8 Circuit Breaker Ved DD-Nivåer

> **"Drawdown-nivåer trigger automatisk nedtrapping"**

| Drawdown | Action |
|----------|--------|
| 5% | Warning, reduce size 25% |
| 8% | Half size, no new positions |
| 12% | Close 50% of positions |
| 15% | Close all, full stop |
| 20% | Kill-switch, 7-day pause |

**Implementation**: `services/risk_kernel/circuit_breakers.py`

---

### §9 Pre-Flight Før All Aktivitet

> **"Ingen trading uten bestått pre-flight checklist"**

Pre-flight must verify:
- System health (all services up)
- Data integrity (no gaps)
- Risk status (within limits)
- Market conditions (tradeable)
- Capital status (sufficient)

**Implementation**: `ops/pre_flight/go_no_go.py`

---

### §10 Kill-Switch Alltid Tilgjengelig

> **"Kill-switch må alltid være tilgjengelig og testet"**

| Kill-Switch Type | Trigger | Effect |
|------------------|---------|--------|
| Manual | Human button | Immediate halt |
| Automatic | System detection | Staged response |
| Emergency | Critical failure | Close all + halt |

**Implementation**: `ops/kill_switch/`

---

### §11 Exit Alltid Tillatt

> **"Exit-ordre må ALDRI blokkeres av systemet"**

Exits have absolute priority:
- Over entry signals
- Over position limits
- Over any other constraint

The only blocked exit: none.

**Implementation**: `services/exit_brain/exit_types.py`

---

### §12 Posisjon = Bevis, Ikke Tro

> **"Systemets posisjon må alltid matche exchange-posisjon"**

| Mismatch | Action |
|----------|--------|
| Minor (< 1%) | Log + reconcile |
| Moderate (1-5%) | Alert + investigate |
| Major (> 5%) | HALT + manual review |

**Implementation**: `services/data_integrity/reconciliation.py`

---

### §13 Slippage Over X = Pause

> **"Unormal slippage trigger trading-pause"**

| Slippage | Action |
|----------|--------|
| < 0.1% | Normal |
| 0.1-0.3% | Warning logged |
| 0.3-0.5% | Pause 1 hour |
| > 0.5% | Pause + review |

**Implementation**: `services/execution/slippage_monitor.py`

---

### §14 Exchange-Ustabil = Flat

> **"Exchange-ustabilitet trigger umiddelbar flat posisjon"**

| Issue | Response |
|-------|----------|
| API latency > 5s | Halt new entries |
| API errors > 3/min | Close positions |
| Exchange maintenance | FLAT + wait |

**Implementation**: `services/market_regime/liquidity_monitor.py`

---

### §15 Logg Alt, Slett Intet

> **"Alle beslutninger og handlinger logges permanent"**

Audit requirements:
- All trades logged
- All decisions logged
- All overrides logged
- Immutable storage
- No deletion possible

**Implementation**: `services/audit_ledger/immutable_store.py`

---

## Decision Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                    DECISION HIERARCHY                        │
├─────────────────────────────────────────────────────────────┤
│  Level 0 │ KILL-SWITCH        │ Immediate halt (VETO)       │
│  Level 1 │ RISK KERNEL        │ Position safety (VETO)      │
│  Level 2 │ POLICY ENGINE      │ Constitutional guard (VETO) │
│  Level 3 │ CAPITAL ALLOCATION │ Resource control            │
│  Level 4 │ EXIT BRAIN         │ Position management         │
│  Level 5 │ ENTRY GATE         │ Trade qualification         │
│  Level 6 │ SIGNAL / AI        │ Advisory only               │
└─────────────────────────────────────────────────────────────┘

VETO Power: Levels 0-2 can reject any action from lower levels.
Advisory Only: Level 6 can suggest but NEVER execute.
```

---

## VETO Chain

When a decision is made:

1. **Signal/AI** proposes action
2. **Entry Gate** qualifies (if entry)
3. **Exit Brain** evaluates (if exit)
4. **Capital Allocation** checks resources
5. **Policy Engine** verifies laws ← **CAN VETO**
6. **Risk Kernel** confirms safety ← **CAN VETO**
7. **Kill-Switch** is always watching ← **CAN VETO**

If ANY level with VETO power rejects → action is cancelled.

---

## Amendment Protocol

Changing a Grunnlov requires:

1. ✅ Written justification
2. ✅ Impact assessment
3. ✅ Shadow testing (30 days minimum)
4. ✅ Unanimous approval
5. ✅ Documented rollback plan
6. ✅ Post-change monitoring (90 days)

**Expected frequency**: Once per year or less.

---

**END OF GOVERNANCE DOCUMENT**
