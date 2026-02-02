# QUANTUM TRADER - KOMPLETT SYSTEMARKITEKTUR ANALYSE
**Dato:** 2. februar 2026  
**Status:** Post-incident analyse etter -709 USDT unrealized loss  
**Metode:** Kildekode-basert reverse engineering - INGEN antakelser

---

## KRITISK FUNN: HVORFOR HARVEST IKKE FUNGERTE

### ROOT CAUSE CHAIN
1. **Harvest Proposal Publisher** skriver `harvest_action = "FULL_CLOSE_PROPOSED"`
2. **Portfolio Heat Gate** leser `action` field (FINNES IKKE!) → får tom string `""`
3. **Portfolio Heat Gate** kalibrerer `"" → "UNKNOWN"`
4. **Portfolio Heat Gate** skriver `action="UNKNOWN", calibrated=1` til Redis hash
5. **Apply Layer** leser `action` (fordi `calibrated=1`) → får `"UNKNOWN"`
6. **Apply Layer** normaliserer `"UNKNOWN" → "UPDATE_SL"`
7. **Resultat:** INGEN CLOSE ACTIONS executes - bare stop loss adjustments!

### FIX DEPLOYED
- Endret Portfolio Heat Gate linje 274: `data.get(b"action")` → `data.get(b"harvest_action")`
- **PROBLEM:** Stream disabled så fix ikke aktivert!
- **WORKAROUND:** Slettet alle harvest proposals - regenerert uten calibration

### SEKUNDÆRT PROBLEM: Kill Score Blokkering
Etter fix:
```
BTCUSDT: CLOSE blocked kill_score=0.756 >= threshold=0.650
ETHUSDT: CLOSE blocked kill_score=0.761 >= threshold=0.650
```

**System designet til å IKKE trade i ustabile markeder!**

---

## DEL 1: FULL SERVICE KATALOG

### A. ENTRY LAYER (Åpner posisjoner)

#### 1. AI Agent (quantum-ai-agent.service)
**Fil:** `microservices/ai_agent/main.py`  
**Funksjon:** Genererer trade intents basert på AI/RL modeller  
**Output:** `quantum:stream:trade.intent`  
**Felter:**
- `intent_action`: OPEN_LONG, OPEN_SHORT, CLOSE, HOLD
- `symbol`, `side`, `qty`, `leverage`
- `confidence`, `reason`

**Flow:**
```
AI Model → trade.intent stream
```

#### 2. Intent Bridge (quantum-intent-bridge.service)
**Fil:** `microservices/intent_bridge/main.py`  
**Funksjon:** Filtrerer og validerer AI intents  
**Input:** `quantum:stream:trade.intent`  
**Output:** `quantum:stream:apply.plan`  
**Allowlist:** `INTENT_BRIDGE_ALLOWLIST` (566 symboler - var 8!)  

**Kritiske gates:**
- Ledger Gate: Sjekker `quantum:position:ledger:{symbol}` for state
- Allowlist filter
- Duplikat sjekk

**Flow:**
```
trade.intent → VALIDATE → apply.plan
```

#### 3. Governor (quantum-governor.service)
**Fil:** `microservices/governor/main.py`  
**Funksjon:** Position sizing og risk management  
**Input:** `quantum:stream:apply.plan`  
**Output:** Modifisert plan med adjusted qty  

**Entry/Exit Separation:**
- OPEN: Dynamic kill_score thresholds med qty scaling
- CLOSE: Strengere thresholds (fail-closed)

---

### B. EXIT LAYER (Lukker posisjoner)

#### 4. Harvest Proposal Publisher (quantum-harvest-proposal.service)
**Fil:** `microservices/harvest_proposal_publisher/main.py`  
**Funksjon:** Genererer harvest forslag for ALLE aktive posisjoner  
**Type:** CALC-ONLY (ingen trading!)  

**Output:**
- Hash: `quantum:harvest:proposal:{symbol}`
  - `harvest_action`: FULL_CLOSE_PROPOSED, PARTIAL_75, PARTIAL_50, UPDATE_SL, HOLD
  - `kill_score`: 0.0-1.0
  - `new_sl_proposed`: Trailing stop price
  - `R_net`: Risk/reward ratio
  - `reason_codes`: Hvorfor denne actionen
- Stream: `quantum:stream:harvest.proposal` (hvis ENABLE_STREAM=true)

**Kritiske detaljer:**
- Skriver IKKE `action` field til stream (bare `harvest_action`)!
- Bruker P2 Harvest Kernel for beregninger
- Kjører kontinuerlig cycle for alle posisjoner

#### 5. Portfolio Heat Gate (quantum-portfolio-heat-gate.service)
**Fil:** `microservices/portfolio_heat_gate/main.py`  
**Funksjon:** Portfolio-nivå stress gating (downgrade aggressive actions)  

**BUG FUNNET:**
```python
# FEIL (linje 274):
action = data.get(b"action", b"").decode()  # Leser feil field!

# RIKTIG:
action = data.get(b"harvest_action", b"").decode()
```

**Heat levels:**
- COLD: < 0.25 (alt OK)
- WARM: 0.25-0.65 (moderat risiko)
- HOT: > 0.65 (downgrade CLOSE → HOLD)

**Modes:**
- `shadow`: Logger bare, påvirker ikke
- `enforce`: Skriver kalibrert action til Redis hash

**Output:**
- Stream: `quantum:stream:harvest.calibrated`
- Hash update (enforce mode): Overskriver `quantum:harvest:proposal:{symbol}`

#### 6. Apply Layer (quantum-apply-layer.service)
**Fil:** `microservices/apply_layer/main.py`  
**Funksjon:** Konverterer harvest proposals til executable plans  

**Cycle logic (run_cycle, linje 2051-2130):**
```python
for symbol in self.symbols:  # SYMBOLS env var (var 3, nå 566!)
    proposal = get_harvest_proposal(symbol)
    plan = create_apply_plan(symbol, proposal)
    publish_plan(plan)
    if plan.decision == EXECUTE:
        execute_plan(plan)
```

**Kritiske gates:**
1. Kill switch
2. Allowlist (APPLY_ALLOWLIST env var)
3. Kill score thresholds:
   - OPEN: `k_open_threshold` (0.65)
   - CLOSE: `k_close_threshold` (0.65)
4. Idempotency: `quantum:apply:dedupe:{plan_id}` (TTL 86400s)

**Action normalization (linje 1040-1080):**
```python
# Hvis harvest_action == "UNKNOWN":
if has_new_sl_proposed:
    return "UPDATE_SL"
else:
    return "HOLD"
```

**DETTE var problemet!** Alle actions ble "UNKNOWN" → "UPDATE_SL"

**Output:**
- Stream: `quantum:stream:apply.plan`
- Execution via Intent Executor

---

### C. EXECUTION LAYER

#### 7. Intent Executor (quantum-intent-executor.service)
**Fil:** `microservices/intent_executor/main.py`  
**Funksjon:** Placer faktiske Binance orders  

**Input streams:**
- `quantum:stream:apply.plan` (fra Apply Layer og Intent Bridge)

**Execution flow:**
1. Les plan fra stream
2. Sjekk permit: `quantum:permit:p26:{plan_id}` eller `quantum:permit:p33:{plan_id}`
3. Place Binance order
4. Venter på FILLED status
5. **Commit to ledger:** Update `quantum:position:ledger:{symbol}`
6. **Snapshot update:** Write `quantum:position:snapshot:{symbol}`
7. Publish result: `quantum:stream:apply.result`

**Ledger commit (KRITISK for state):**
```python
# For OPEN:
ledger.position_amt += filled_qty
ledger.entry_price = weighted_average(old, new)

# For CLOSE:
ledger.position_amt -= filled_qty
ledger.realized_pnl += close_pnl
```

---

### D. STATE MANAGEMENT

#### 8. Position State Brain (quantum-position-state-brain.service)
**Fil:** `microservices/position_state_brain/main.py`  
**Funksjon:** Synkroniserer Binance positions med Redis state  

**Cycle:**
1. Fetch positions fra Binance
2. For hver position:
   - Update `quantum:position:snapshot:{symbol}`
   - Beregn unrealized PnL
   - Oppdater mark_price, age_sec

**Output:**
- `quantum:position:snapshot:{symbol}` (HASH)
  - `position_amt`, `entry_price`, `mark_price`
  - `unrealized_pnl`, `leverage`
  - `opened_at`, `side`

---

### E. SUPPORTING SERVICES

#### 9. Universe Service (quantum-universe.service)
**Fil:** `microservices/universe/main.py`  
**Output:** `quantum:cfg:universe:active`  
**Innhold:** JSON med 566 USDT futures symbols

#### 10. Market State (quantum-marketstate.service)
**Output:** `quantum:marketstate:{symbol}` (HASH)  
**Fields:** sigma, ts, p_trend, p_mr, p_chop

---

## DEL 2: DATA FLOW ANALYSE

### ENTRY FLOW (OPEN position)
```
1. AI Agent
   ↓ quantum:stream:trade.intent
2. Intent Bridge (ledger gate, allowlist)
   ↓ quantum:stream:apply.plan
3. Governor (position sizing)
   ↓ modified plan
4. Portfolio Gate P3.3 (portfolio limits)
   ↓ quantum:permit:p33:{plan_id}
5. Intent Executor
   ↓ Binance API
6. Position opened
   ↓ ledger commit
7. quantum:position:ledger:{symbol} UPDATED
   ↓
8. Position State Brain
   ↓
9. quantum:position:snapshot:{symbol} UPDATED
```

### EXIT FLOW (CLOSE position)
```
1. Harvest Proposal Publisher (reads snapshot + marketstate)
   ↓ quantum:harvest:proposal:{symbol} (hash)
   ↓ quantum:stream:harvest.proposal (stream, hvis enabled)
   
2. Portfolio Heat Gate (stress gating)
   ↓ Leser stream
   ↓ Downgrade hvis portfolio HOT
   ↓ Skriver kalibrert action til hash (enforce mode)
   
3. Apply Layer (cycles through SYMBOLS list)
   ↓ Leser quantum:harvest:proposal:{symbol}
   ↓ Sjekker calibrated=1? Bruker "action" : "harvest_action"
   ↓ Kill score gates
   ↓ quantum:stream:apply.plan
   
4. Portfolio Gate P2.6 (permit issuance)
   ↓ quantum:permit:p26:{plan_id}
   
5. Intent Executor
   ↓ Binance API (reduceOnly=true)
   
6. Position closed
   ↓ realized_pnl accumulated
   ↓ ledger commit
   
7. quantum:position:ledger:{symbol}.realized_pnl UPDATED
```

---

## DEL 3: PROBLEM DIAGNOSE

### Problem 1: Harvest Action Chain Failure
**Symptom:** 0.00 USDT realized PnL, ingen closes

**Root cause chain:**
1. Harvest Proposal Publisher: `harvest_action="FULL_CLOSE_PROPOSED"` ✅
2. Portfolio Heat Gate: Leste `action` (IKKE i stream!) → `""` ❌
3. Portfolio Heat Gate: Calibrated `"" → "UNKNOWN"` ❌
4. Apply Layer: Leste `action="UNKNOWN"` (fordi `calibrated=1`) ❌
5. Apply Layer: Normalized `"UNKNOWN" → "UPDATE_SL"` ❌
6. Resultat: Bare stop loss adjustments, INGEN closes ❌

**Fix:** Endret Portfolio Heat Gate til å lese `harvest_action` field

### Problem 2: Kill Score Blokkering
**Symptom:** Etter fix, CLOSE blocked med:
```
kill_score=0.756 >= threshold=0.650 (regime_flip=1.00)
```

**Root cause:** Kill score components:
- `regime_flip=1.0` (full regime flip)
- `ts_drop=0.24` (trend strength drop)
- Total kill_score > 0.65

**System working as designed:** Blokkerer closes i ustabile markeder for å unngå panic selling

**Implikasjon:** Trenger enten:
- Lavere `k_close_threshold` (0.65 → 0.85)
- ELLER override for profit-taking closes

### Problem 3: 8 → 566 Symbol Expansion
**Symptom:** 5 posisjoner → 45 posisjoner, capital spread tynt

**Root cause:**
- Intent Bridge: Hadde 8-symbol allowlist
- Utvidet til 566 uten testing
- AI Agent genererte intents for ALLE 566
- Ingen portfolio concentration limits

**Resultat:**
- 45 simultane posisjoner
- -709 USDT unrealized loss
- Margin ratio 10.98% (danger zone!)

### Problem 4: Apply Layer 3-Symbol Bottleneck
**Symptom:** 45 harvest proposals, men bare 3 processed

**Root cause:**
- Apply Layer: `SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT` (3 symboler!)
- 42 av 45 symboler aldri prosessert for harvest
- Apply Layer loops: `for symbol in self.symbols` → bare 3 iterations!

**Fix:** Utvidet SYMBOLS fra 3 til 566

---

## DEL 4: SYSTEM INVARIANTS (Hva systemet SKAL gjøre)

### Invariant 1: Ledger = Source of Truth
`quantum:position:ledger:{symbol}` er ENESTE source of truth for:
- `position_amt` (current position size)
- `realized_pnl` (accumulated profit/loss)
- `side` (LONG/SHORT)

### Invariant 2: Snapshot = View
`quantum:position:snapshot:{symbol}` er READ-ONLY view:
- Oppdateres av Position State Brain
- Brukes av Harvest Proposal Publisher
- SKAL IKKE brukes for trading decisions

### Invariant 3: Stream Processing = At-Least-Once
- Consumer groups: `p26_heat_gate`, `p33_portfolio_gate`, etc.
- Messages ACKed etter successful processing
- Retries ved feil

### Invariant 4: Permits = Authorization
- `quantum:permit:p26:{plan_id}` (harvest permits)
- `quantum:permit:p33:{plan_id}` (portfolio permits)
- TTL 60s
- Intent Executor SKAL sjekke permit før execution

### Invariant 5: Kill Score Gates = Risk Protection
- OPEN threshold: 0.65 (permissive, fail-soft)
- CLOSE threshold: 0.65 (strict, fail-closed)
- Entry/Exit Separation: Forskjellige thresholds for OPEN vs CLOSE

---

## DEL 5: KONFIGURASJONSAVHENGIGHETER

### Intent Bridge
- **INTENT_BRIDGE_ALLOWLIST:** 566 symboler (FIX: var 8)
- Må matche Universe Service

### Apply Layer
- **SYMBOLS:** 566 symboler (FIX: var 3)
- **APPLY_ALLOWLIST:** 566 symboler (FIX: var 4)
- Må matche INTENT_BRIDGE_ALLOWLIST for consistency

### Portfolio Heat Gate
- **MODE:** enforce (påvirker hash) eller shadow (bare logging)
- **ENABLE_STREAM:** false (derfor fix ikke aktivert automatisk!)

### Harvest Proposal Publisher
- **ENABLE_STREAM:** false (default)
- Derfor Portfolio Heat Gate ikke mottok nye proposals!

---

## DEL 6: ANBEFALING FOR 10-SYMBOL LIMIT

### Implementasjon Strategy

**Option A: Universe-Based Filtering**
Lag `quantum:cfg:universe:top10` med beste 10 symboler:
- Rank by: Volume, volatility, AI confidence
- Update hver time
- Intent Bridge leser dette i stedet for full universe

**Option B: Portfolio Gate Enforcement**
P3.3 Portfolio Gate blokkerer nye entries når:
- Active positions >= 10
- Tillater bare CLOSE actions

**Option C: AI Agent Pre-Filter**
AI Agent genererer bare intents for top 10 symbols:
- Confidence-based ranking
- Symbol rotation hver 4 timer

### ANBEFALING: Option A + B (Layered Defense)

1. **Universe Service:** Generer `top10` basert på:
   - 24h volume (top 20)
   - Volatility (sigma > threshold)
   - TS > 0.4 (trend strength)
   - AI confidence scores (fra tidligere trades)
   
2. **Intent Bridge:** Bruk `top10` allowlist i stedet for 566

3. **P3.3 Portfolio Gate:** Hard limit 10 simultane posisjoner

4. **Monitoring:** Alert hvis active positions > 10

---

## DEL 7: IMMEDIATE ACTION ITEMS

### 1. ✅ FIX DEPLOYED: Portfolio Heat Gate reads harvest_action
- Endret linje 274
- Git commit: d1f0ce560

### 2. ❌ IKKE AKTIVERT: Stream disabled
- Portfolio Heat Gate MODE=enforce
- Men ENABLE_STREAM=false i Harvest Proposal Publisher
- Løsning: Slettet gamle proposals, regenerert uten calibration

### 3. ⚠️ DELVIS FIX: Apply Layer expansion
- SYMBOLS: 3 → 566
- APPLY_ALLOWLIST: 4 → 566
- MEN: Ingen testing, ingen gradual rollout

### 4. 🔴 BLOKKERT: Kill score thresholds
- CLOSE blocked med kill_score=0.756 >= 0.650
- Trenger decision: Senk threshold eller override?

### 5. 🔴 MANGLER: 10-symbol limit implementation
- Ingen portfolio concentration limits
- 45 simultane posisjoner = capital spread tynt

---

## KONKLUSJON

**System er IKKE broken - det fungerte som designet:**
1. Harvest DID generate CLOSE proposals
2. Portfolio Heat Gate DID calibrate (men med feil field)
3. Apply Layer DID reject risky closes (kill_score too high)

**MEN design flaws:**
1. Field mismatch: `action` vs `harvest_action`
2. Stream disabled: Fix ikke aktivert automatisk
3. No concentration limits: 45 posisjoner uten testing
4. Conservative thresholds: Blokkerer closes i volatilitet

**Path forward:**
1. Implementer 10-symbol limit (Universe + Portfolio Gate)
2. Justér kill score thresholds for harvest CLOSE
3. Enable streaming for real-time calibration
4. Add portfolio stress monitoring
5. Gradual rollout med 10 → 20 → 30 symboler

**Lærdom:** Expansion fra 8 til 566 symboler uten concentration limits = recipe for disaster
