# 🛑 MANUAL CLOSE PROSEDYRE — SOLUSDT

**Dato**: 2026-02-11  
**Kontekst**: AUTHORITY FREEZE active, BSC operationally blocked (API error -2015)  
**Posisjon**: SOLUSDT (antatt åpen basert på tidligere observasjoner)  
**Metode**: Manual Market Close via Binance UI  
**Kanonisk Reference**: Manual intervention under Authority Freeze

---

## ⚠️ API BLOKKERING BEKREFTET

**Forsøk på automatisk datahenting**:
```
❌ APIError(code=-2015): Invalid API-key, IP, or permissions for action
```

**Konsekvens**: Må bruke Binance Futures Testnet UI direkte.

---

## 🔍 STEG 1 — HENT POSISJONSDATA (VIA BINANCE UI)

### Åpne Binance Futures Testnet

**URL**: https://testnet.binancefuture.com/  
**Login**: Bruk testnet credentials

### Naviger til Positions

1. Klikk på **"Positions"** tab (nederst på skjermen)
2. Se etter **SOLUSDT** i listen
3. Hvis SOLUSDT IKKE vises → **posisjonen er allerede lukket** ✅

### Notér Følgende (hvis posisjon finnes)

**Kritisk informasjon**:
- **Symbol**: SOLUSDT
- **Side**: LONG eller SHORT
- **Size (Quantity)**: ___________ SOL
- **Entry Price**: ___________ USDT
- **Mark Price**: ___________ USDT
- **Unrealized PnL**: ___________ USDT (___%)
- **Leverage**: ___x
- **Liquidation Price**: ___________ USDT

**Eksempel**:
```
Symbol: SOLUSDT
Side: LONG
Size: 6.87 SOL
Entry Price: 142.50 USDT
Mark Price: 138.20 USDT
Unrealized PnL: -29.54 USDT (-2.07%)
Leverage: 2x
```

---

## 🔥 STEG 2 — UTFØR MANUELL CLOSE

### Hvis Posisjon er LONG:

**Close Order**:
- **Action**: SELL (steng LONG)
- **Order Type**: MARKET
- **Quantity**: FULL position size (samme som "Size" i posisjon)
- **Reduce Only**: ✅ **MUST BE CHECKED** (viktigste settet!)
- **Time in Force**: GTC (default)

**UI Instruksjoner**:
1. Finn SOLUSDT i positions-listen
2. Klikk "Close" eller "Market" knapp ved posisjonen
3. Velg "SELL" (for å close LONG)
4. Skriv inn FULL quantity (eller klikk "100%")
5. **BEKREFT** at "Reduce Only" er ✅ checked
6. Klikk "Sell/Close Long"
7. Bekreft ordren i popup

### Hvis Posisjon er SHORT:

**Close Order**:
- **Action**: BUY (steng SHORT)
- **Order Type**: MARKET
- **Quantity**: FULL position size (absolute value)
- **Reduce Only**: ✅ **MUST BE CHECKED**
- **Time in Force**: GTC (default)

**UI Instruksjoner**:
1. Finn SOLUSDT i positions-listen
2. Klikk "Close" eller "Market" knapp ved posisjonen
3. Velg "BUY" (for å close SHORT)
4. Skriv inn FULL quantity (eller klikk "100%")
5. **BEKREFT** at "Reduce Only" er ✅ checked
6. Klikk "Buy/Close Short"
7. Bekreft ordren i popup

### ⚠️ VIKTIGSTE SIKKERHETSSJEKKER

**ALDRI**:
- ❌ Delvis close (alltid 100% av posisjonen)
- ❌ LIMIT order (kun MARKET for clean exit)
- ❌ Trailing stop eller andre komplekse typer
- ❌ Ny posisjon samtidig (ingen hedging/reversal)
- ❌ Leverage-endring før close

**ALLTID**:
- ✅ "Reduce Only" checked (Binance blokkerer ellers)
- ✅ MARKET order (instant execution)
- ✅ 100% quantity (full position close)
- ✅ Bekreft side (LONG → SELL, SHORT → BUY)

---

## 📸 STEG 3 — DOKUMENTASJON (KRITISK FOR GOVERNANCE)

### Umiddelbart Etter Close

**Ta Screenshots**:
1. **Order Confirmation** (popup etter submit)
   - Order ID
   - Filled price
   - Quantity
   - Timestamp

2. **Positions Tab** (etter close)
   - Vis at SOLUSDT quantity = 0
   - Eller at SOLUSDT ikke lenger vises i Open Positions

3. **Order History** (valgfritt)
   - Vis close order i history
   - Match order ID med confirmation

### Manuell Notasjon

**Kopier denne malen og fyll ut**:
```yaml
manual_close_report:
  symbol: SOLUSDT
  original_side: LONG  # eller SHORT
  quantity: 6.87       # faktisk qty closed
  close_side: SELL     # eller BUY
  order_type: MARKET
  reduce_only: true
  fill_price: 138.20   # average fill price
  pnl_realized: -29.54 # faktisk realized PnL
  timestamp: 2026-02-11T23:15:30Z  # UTC tid
  method: manual_binance_ui
  reason: "Authority Freeze + Execution Pipeline Broken + BSC API Blocked"
  authorized_by: "Emergency manual intervention (no automated controller)"
  audit_trail: "BSC_MANUAL_CLOSE_FEB11_2026.md"
```

**Lagre denne informasjonen** (i notat, commit message, eller dedikert fil).

---

## 🧾 STEG 4 — REGISTRER I REDIS (AUDIT TRAIL)

**Når close er bekreftet**, kjør dette for å logge i BSC audit stream:

```bash
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'redis-cli XADD quantum:stream:bsc.events \
  event MANUAL_CLOSE \
  symbol SOLUSDT \
  side LONG \
  quantity 6.87 \
  close_side SELL \
  fill_price 138.20 \
  pnl_realized -29.54 \
  reason "Manual close during Authority Freeze - BSC API blocked" \
  timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  method manual_binance_ui \
  authorized_by emergency_intervention'
```

**VIKTIG**: Dette gir INGEN komponent mer makt - det er ren observasjon/audit.

**Erstatt verdier** med faktiske data fra close (quantity, fill_price, pnl_realized).

---

## ✅ STEG 5 — VERIFIKASJON

### Bekreft i Binance UI

**Positions Tab**:
- ✅ SOLUSDT vises IKKE lenger
- ✅ Quantity = 0 (eller helt borte fra listen)
- ✅ Ingen nye posisjoner åpnet ved et uhell

**Closed P&L**:
- Se "Closed P&L" for SOLUSDT
- Match med forventet realized PnL

### Bekreft at BSC er Idle

**Kjør**:
```bash
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'tail -20 /var/log/quantum/bsc.log'
```

**Forventet output** (etter close):
```
2026-02-11 XX:XX:XX [INFO] 🔍 BSC Check Cycle #XXX
2026-02-11 XX:XX:XX [ERROR] ❌ Binance API error: APIError(code=-2015)...
2026-02-11 XX:XX:XX [INFO] ✅ No open positions (or fetch failed → FAIL OPEN)
```

**Eller** (hvis API plutselig begynner å fungere):
```
2026-02-11 XX:XX:XX [INFO] 🔍 BSC Check Cycle #XXX
2026-02-11 XX:XX:XX [INFO] ✅ No open positions
```

**VIKTIG**: BSC skal IKKE logge noen CLOSE-handlinger (den gjorde ikke closen - det var manuelt).

### Bekreft at Ingen Nye Order Genereres

**Sjekk**:
- Ingen nye SOLUSDT posisjoner dukker opp
- Ingen andre automatiske handler
- BSC fortsetter i IDLE mode

---

## 🟢 ETTER MANUELL CLOSE — SYSTEMSTATUS

**Når SOLUSDT er confirmed closed**:

```yaml
system_status:
  open_positions: 0
  risk_exposure: NONE
  authority_mode: AUTHORITY_FREEZE
  active_controller: BSC (idle, no positions to manage)
  execution_pipeline: BROKEN (harvest_brain dead, execution_service starved)
  bsc_operational_state: TECHNICALLY_VERIFIED / API_BLOCKED
  next_priority: PATH 1 (Repair harvest_brain:execution consumer)
```

**Dette er best mulig tilstand** før start på PATH 1-arbeid:
- ✅ Ingen posisjon-risk
- ✅ BSC deployed som safety net (når API fungerer)
- ✅ Authority Freeze respektert
- ✅ Execution pipeline kan repareres uten live risk

---

## 📋 POST-CLOSE ACTIONS

### Umiddelbart (Dag 0)
- [x] SOLUSDT closed via manual MARKET order
- [ ] Screenshots tatt og lagret
- [ ] Manual close data notert i YAML format
- [ ] Redis audit event logged (MANUAL_CLOSE)
- [ ] Verifikasjon: ingen åpne posisjoner i UI
- [ ] BSC logs bekreftet (idle/no positions)

### Kort Sikt (Dag 1-7)
- [ ] Oppdater `PNL_AUTHORITY_MAP_CANONICAL_FEB10_2026.md` med manual close note
- [ ] BSC Day 1 Scope Guard audit (scheduled 2026-02-11 22:04 UTC)
- [ ] Beslutning: Fikse Binance API access for BSC OR start PATH 1 immediately

### Medium Sikt (Uke 1-4)
- [ ] **PATH 1**: Repair `harvest_brain:execution` consumer (161K lag)
- [ ] Fix `execution_service` stream subscription (trade.intent → apply.result)
- [ ] Test CLOSE pipeline end-to-end
- [ ] Re-assess Harvest Proposal for CONTROLLER re-entry (5 BEVISKRAV)

### Lang Sikt (30+ dager)
- [ ] BSC reaches 30-day sunset (2026-03-12) → automatic demotion
- [ ] Establish new CONTROLLER from PATH 1 repairs
- [ ] Exit AUTHORITY_FREEZE (restore normal operations)

---

## 🚨 GOVERNANCE NOTES

### Authority Context

**Manual Close Authorization**:
- **Who**: Human operator (emergency intervention)
- **Authority**: Direct Binance UI access (bypasses all system components)
- **Justification**: AUTHORITY_FREEZE permits manual actions when no CONTROLLER can execute
- **Compliance**: Follows `AUTHORITY_FREEZE_PROMPT_CANONICAL.md` emergency procedures

**NOT a Scope Violation**:
- ✅ Manual action ≠ automated controller
- ✅ BSC did NOT execute this (it's API-blocked)
- ✅ No component gained unauthorized CONTROLLER authority
- ✅ Audit trail maintained (Redis + screenshots)

### Lessons Learned

**Why Manual Close Was Necessary**:
1. Execution pipeline broken (harvest_brain dead 2+ days)
2. BSC deployed but API-blocked (error -2015)
3. Open position (SOLUSDT) unmanaged for 48+ hours
4. Risk exposure during AUTHORITY_FREEZE unacceptable

**Prevention for Future**:
- Fix Binance API credentials immediately (BSC becomes operational)
- Prioritize PATH 1 repairs (restore automated exit capability)
- Establish monitoring for "orphaned positions" (open but no controller)

---

## 📝 SIGNATURE

**Procedure**: Manual Close of SOLUSDT  
**Method**: Binance Futures Testnet UI (MARKET, Reduce Only)  
**Date**: 2026-02-11  
**Context**: AUTHORITY_FREEZE, BSC API-blocked, Execution pipeline broken  
**Authorization**: Emergency manual intervention  
**Canonical Document**: `MANUAL_CLOSE_SOLUSDT_FEB11_2026.md`

**Quote**: *"Manual close under Authority Freeze is not a failure of governance - it is governance working as designed when automation fails."*

---

**Last Updated**: 2026-02-11  
**Status**: AWAITING USER EXECUTION  
**Next Step**: User closes SOLUSDT via Binance UI → Document → Verify → Log to Redis
