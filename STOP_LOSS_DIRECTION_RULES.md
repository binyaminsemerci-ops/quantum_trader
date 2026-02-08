# Stop Loss Direction Logic - Kritisk Dokumentasjon

**⚠️ CRITICAL: Les dette før du jobber med SL/TP beregninger!**

## 🎯 Grunnleggende Regel

**SHORT og LONG har MOTSATT SL-retning:**

### LONG Positions (BUY)
- **Stop Loss:** UNDER entry price
- **Take Profit:** OVER entry price
- **Logikk:** Beskytter mot pris fall, tar profit på pris økning

```python
# LONG (BUY) - RIKTIG
entry = 100.0
stop_loss = entry * (1 - 0.02)  # 98.0 (2% UNDER) ✅
take_profit = entry * (1 + 0.02)  # 102.0 (2% OVER) ✅
```

### SHORT Positions (SELL)
- **Stop Loss:** OVER entry price  
- **Take Profit:** UNDER entry price
- **Logikk:** Beskytter mot pris økning, tar profit på pris fall

```python
# SHORT (SELL) - RIKTIG
entry = 100.0
stop_loss = entry * (1 + 0.02)  # 102.0 (2% OVER) ✅
take_profit = entry * (1 - 0.02)  # 98.0 (2% UNDER) ✅
```

## 🚨 Historisk Bug - Aldri Gjenta!

**2026-02-07: Oppdaget systemisk feil** hvor ALL SHORT positions hadde SL i feil retning:

```python
# ❌ FEIL - Dette skjedde i production:
AAVEUSDT SHORT:
  Entry: 103.15
  SL (Feil): 99.96  # UNDER entry - kun trigger på downside (god retning!)
  SL (Riktig): 106.34  # OVER entry - trigger på upside (beskytter mot tap!)

Resultat: Position tapte -12.48 USDT uten å trigger SL
```

**Root Cause:** `trading_bot/simple_bot.py` brukte hardkodet LONG-logikk for defaults:
```python
# BUG (fixed):
"stop_loss": prediction.get("stop_loss", market_data["price"] * 0.98)  # ALWAYS below!
"take_profit": prediction.get("take_profit", market_data["price"] * 1.02)  # ALWAYS above!
```

## ✅ Korrekt Implementasjon

### 1. Trading Bot Defaults
```python
# trading_bot/simple_bot.py - Lines ~325-340
if action == "BUY":
    default_sl = price * (1 - sl_percent)  # BELOW
    default_tp = price * (1 + tp_percent)  # ABOVE
elif action == "SELL":
    default_sl = price * (1 + sl_percent)  # ABOVE ✅ 
    default_tp = price * (1 - tp_percent)  # BELOW ✅
```

### 2. Apply Layer Validation
```python
# apply_layer/main.py - Lines ~2598-2620
if position_side == "SHORT" and sl_float < entry_float:
    logger.error(f"❌ INVALID SL for SHORT - correcting")
    sl_validated = str(entry_float * 1.02)  # Force above
elif position_side == "LONG" and sl_float > entry_float:
    logger.error(f"❌ INVALID SL for LONG - correcting")
    sl_validated = str(entry_float * 0.98)  # Force below
```

### 3. Harvest Brain Emergency Trigger
```python
# harvest_brain/harvest_brain.py - Lines ~487-533
if position.side == 'LONG' and position.current_price <= position.stop_loss:
    sl_triggered = True
elif position.side == 'SHORT' and position.current_price >= position.stop_loss:
    sl_triggered = True  # Triggers when price goes UP
```

## 🧪 Obligatoriske Tester

**Kjør alltid tester før deploy:**
```bash
pytest tests/test_stop_loss_direction.py -v
```

Testene sikrer:
- ✅ LONG SL alltid UNDER entry
- ✅ SHORT SL alltid OVER entry
- ✅ Validation fanger opp feil
- ✅ Real-world cases (AAVEUSDT bug) fanges

## 📋 Deployment Checklist

Når du jobber med SL/TP logic, ALLTID:

1. [ ] Sjekk at `side` eller `action` er korrekt identifisert
2. [ ] Verifiser at kondisjonalen `if side == "LONG" vs "SHORT"` brukes
3. [ ] Test med faktiske verdier: 
   - LONG: entry=100, SL må være <100
   - SHORT: entry=100, SL må være >100
4. [ ] Kjør enhetstester
5. [ ] Deploy til testnet FØRST
6. [ ] Verifiser i Redis: `redis-cli hgetall quantum:position:SYMBOL`
7. [ ] Sjekk logger for validation errors

## 🔍 Debugging

**Symptomer på feil SL-retning:**
- SHORT positions taper stort uten å lukke
- SL trigger kun når prisen går i "god" retning for position
- Redis position data: SHORT med SL < entry

**Diagnostisering:**
```bash
# Sjekk faktiske SL verdier
redis-cli hgetall quantum:position:SYMBOLUSDT | grep -E "(side|entry_price|stop_loss)"

# Sjekk SL trigger logg
tail -f /var/log/quantum/harvest_brain.log | grep "SL_TRIGGERED|EMERGENCY"
```

## ⚡ Quick Reference

| Position | Entry | Correct SL | Wrong SL |
|----------|-------|------------|----------|
| LONG     | 100   | 98 (below) | 102 (above) |
| SHORT    | 100   | 102 (above) | 98 (below) |

**Husk:** SHORT profiterer når pris går NED, men må beskyttes mot pris som går OPP!

---

**Sist oppdatert:** 2026-02-07  
**Oppdatert av:** Emergency Fix (SHORT SL bug)  
**Related:** AI_STOP_LOSS_TRIGGER_FIX_FEB6.md, harvest_brain.py
