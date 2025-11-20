# 🎯 Position Sizing Fix - 30x Leverage med 4 Posisjoner

**Dato:** 20. november 2025  
**Problem:** AI-en kjøpte alt for små posisjoner selv om vi bare hadde 4 par aktive  
**Root Cause:** Feil forståelse av margin vs position size, og feil % allokering

---

## 🔍 Problembeskrivelse

### Symptomer
- AI-en åpnet posisjoner som var alt for små i forhold til balansen
- Selv med bare 4 aktive posisjoner ble ikke kapitalen effektivt utnyttet
- 30x leverage ble ikke ordentlig implementert i position sizing

### Root Cause Analyse

**3 kritiske feil ble identifisert:**

1. **Feil forståelse av `max_notional_per_trade`**
   - Koden behandlet dette som position size
   - Det skulle være **MARGIN** (collateral), ikke position size
   - Med 30x leverage: $5000 margin = $150,000 position size

2. **Feil cash allocation per trade**
   - Event-driven executor brukte 95% av cash per trade
   - Dette er umulig med flere posisjoner
   - Med 4 posisjoner burde hver få 25% av tilgjengelig margin

3. **Inkonsistent beregning mellom modules**
   - `event_driven_executor.py`: Brukte gammel notional-basert beregning
   - `autonomous_trader.py`: Hadde komplisert risk-based beregning som delte på leverage (feil retning)
   - `market_config.py`: FUTURES hadde kun 2% max_position_size

---

## ✅ Løsning Implementert

### 1. Event-Driven Executor (`backend/services/event_driven_executor.py`)

**FØR:**
```python
cash = await self._adapter.get_cash_balance()
max_notional = self._risk_config.max_notional_per_trade or 4000.0
notional = min(max_notional, cash * 0.95)  # 95% of cash!
quantity = notional / price
```

**ETTER:**
```python
cash = await self._adapter.get_cash_balance()
max_margin_per_trade = self._risk_config.max_notional_per_trade or 5000.0
margin_from_balance = cash * 0.25  # 25% per trade (4 positions)
actual_margin = min(max_margin_per_trade, margin_from_balance)

leverage = 30
position_size_usd = actual_margin * leverage  # $5000 margin * 30 = $150k position
quantity = position_size_usd / price
```

**Resultat:**
- Bruker 25% av cash som margin per trade (4 posisjoner)
- Multipliserer margin med 30x leverage for å få faktisk position size
- Klar logging som viser margin, position size og leverage

### 2. Autonomous Trader (`backend/trading_bot/autonomous_trader.py`)

**FØR:**
```python
risk_amount = available_balance * max_position_percent * confidence_multiplier
effective_risk = risk_amount / leverage  # FEIL! Delte på leverage
position_size = effective_risk / (price * stop_loss_percent)
```

**ETTER:**
```python
margin = available_balance * max_position_percent * confidence_multiplier
position_size_usd = margin * leverage  # MULTIPLISERER med leverage
quantity = position_size_usd / price
```

**Resultat:**
- Margin multipliseres med leverage (ikke deles)
- Direkte konvertering fra USD position size til quantity
- Enklere og mer forståelig kode

### 3. Market Config (`backend/trading_bot/market_config.py`)

**FØR:**
```python
"FUTURES": {
    "max_position_size": 0.02,  # 2% av portfolio
}
```

**ETTER:**
```python
"FUTURES": {
    "max_position_size": 0.25,  # 25% av portfolio MARGIN (4 max positions)
}
```

**Resultat:**
- Konsistent med 4 maksimale posisjoner (100% / 4 = 25%)
- Tydelig kommentar om at dette er MARGIN, ikke position size

### 4. Environment Configuration (`.env`)

**FØR:**
```bash
QT_MAX_POSITIONS=6
QT_MAX_NOTIONAL_PER_TRADE=5000.0   # $600 per trade margin = $18,000 position
```

**ETTER:**
```bash
QT_MAX_POSITIONS=4                     # Max 4 concurrent positions (25% margin each)
QT_MAX_NOTIONAL_PER_TRADE=5000.0      # MAX $5000 MARGIN per trade = $150k position @ 30x
QT_MAX_GROSS_EXPOSURE=150000.0        # 4 positions x $37,500 = $150k total exposure
QT_MAX_POSITION_PER_SYMBOL=40000.0    # Up to $40k position per symbol
```

**Resultat:**
- Tydelig dokumentasjon av hva hver variabel betyr
- Korrekt beregning av total exposure
- Max 4 posisjoner eksplisitt satt

---

## 📊 Beregningseksempel

### Med $10,000 balanse og 4 posisjoner:

| Parameter | Verdi | Forklaring |
|-----------|-------|------------|
| **Total Balance** | $10,000 | Tilgjengelig cash på Binance Futures |
| **Max Positions** | 4 | Maksimalt antall samtidige posisjoner |
| **Margin per Position** | $2,500 | $10,000 × 25% = $2,500 margin/collateral |
| **Leverage** | 30x | Binance Futures leverage |
| **Position Size** | $75,000 | $2,500 margin × 30 = $75,000 faktisk posisjon |
| **Total Exposure** | $300,000 | 4 × $75,000 = $300,000 total exposure |

### Med $1,500 balanse (din nåværende):

| Parameter | Verdi | Forklaring |
|-----------|-------|------------|
| **Total Balance** | $1,500 | Nåværende tilgjengelig cash |
| **Max Positions** | 4 | Maksimalt antall samtidige posisjoner |
| **Margin per Position** | $375 | $1,500 × 25% = $375 margin/collateral |
| **Leverage** | 30x | Binance Futures leverage |
| **Position Size** | $11,250 | $375 margin × 30 = $11,250 faktisk posisjon |
| **Total Exposure** | $45,000 | 4 × $11,250 = $45,000 total exposure |

**MEN:** Siden `QT_MAX_NOTIONAL_PER_TRADE=5000.0`, vil den bruke minimum av:
- $375 (25% av balance)
- $5000 (konfigurert max)

Så med $1,500 balance vil hver trade bruke **$375 margin = $11,250 position** per trade.

---

## 🎯 Forventet Atferd Etter Fix

### Når AI åpner ny posisjon:

1. **Check balance:** Henter tilgjengelig cash fra Binance
2. **Calculate margin:** 25% av balance (max 4 posisjoner)
3. **Apply limit:** Bruker min(calculated_margin, $5000)
4. **Apply leverage:** Multipliserer margin med 30x
5. **Calculate quantity:** position_size / current_price
6. **Log details:** Viser margin, position size, leverage i logs

### Logger du vil se:

```
💰 Cash: $1500.00, Margin per trade: $375.00 (max $5000.00, 25% of balance = $375.00)
📤 Placing BUY order: BTCUSDT qty=0.1250 @ $90000.00 (margin=$375.00, position=$11250.00 @ 30x, conf=75.00%)
```

---

## 🚀 Testing Checklist

- [ ] **Backend restart**: Restart backend for å laste nye konfigurasjoner
- [ ] **Check logs**: Verifiser at "margin=$XXX, position=$YYY @ 30x" vises i logs
- [ ] **Verify calculations**: 
  - margin × 30 = position size
  - position size / price = quantity
- [ ] **Check Binance orders**: Verifiser at order størrelser er korrekte på Binance
- [ ] **Monitor 4 positions**: Sikre at AI ikke åpner mer enn 4 posisjoner samtidig
- [ ] **Check P&L**: Med større posisjoner vil P&L bevege seg raskere

---

## ⚠️ Viktige Merknader

### Risk Management
Med større posisjoner kommer større risk:
- **Stop Loss** aktiverer raskere (3% margin loss = force close)
- **Dynamic TP/SL** vil justere oftere
- **Liquidation risk** øker med høyere leverage

### Recommendations
1. **Start med paper trading** (`QT_PAPER_TRADING=true`)
2. **Monitor første 24 timer** nøye
3. **Sjekk emergency SL** fungerer ved -3% margin loss
4. **Verifiser TP/SL orders** settes korrekt

---

## 📝 Files Changed

### Modified Files
1. `backend/services/event_driven_executor.py`
   - Ny margin-basert beregning
   - 25% cash allocation per trade
   - Tydelig logging med margin/position/leverage

2. `backend/trading_bot/autonomous_trader.py`
   - Omskrevet `_calculate_position_size()`
   - Margin × leverage (ikke margin / leverage)
   - Enklere og mer forståelig logikk

3. `backend/trading_bot/market_config.py`
   - FUTURES max_position_size: 2% → 25%
   - Oppdaterte kommentarer

4. `.env`
   - QT_MAX_POSITIONS: 6 → 4
   - Oppdatert dokumentasjon av alle variabler
   - Tydeliggjort margin vs position size

### New Files
- `POSITION_SIZING_FIX_20251120.md` (denne filen)

---

## 🎓 Takeaways

### For fremtidige endringer:
1. **Alltid skille mellom margin og position size** når leverage er involvert
2. **Margin er collateral** - det du setter inn
3. **Position size er eksponering** - margin × leverage
4. **Med N posisjoner: bruk ~(100/N)% margin per posisjon**
5. **Test alltid i paper trading først** når position sizing endres

### Nøkkelformel for leverage trading:
```
margin = balance × allocation_pct
position_size = margin × leverage
quantity = position_size / price
```

---

## ✅ Status: FIXED

Alle endringer er implementert og klare for testing. Restart backend og observer logs for å verifisere korrekt atferd.

**Neste steg:**  
1. Commit endringer til Git
2. Restart backend service
3. Verifiser logging
4. Test med paper trading først
5. Overvåk første live trades nøye
