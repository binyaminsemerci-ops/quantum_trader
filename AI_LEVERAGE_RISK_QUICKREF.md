# 🛡️ Leverage-Aware Risk Management - Quick Reference

## 📋 TL;DR

Exit Brain V3 now **automatically limits maximum margin loss per trade** regardless of leverage.

**Default:** Max 10% margin loss per trade  
**Formula:** `allowed_move_pct = 0.10 / leverage`  
**Action:** Tightens SL if AI sets risky stop OR force-exits if over-leveraged

---

## ⚡ Quick Examples

### Example 1: 5x Leverage
```
Entry: $50,000
Leverage: 5x
Max margin loss: 10%

allowed_move_pct = 10% / 5 = 2.0%
Risk floor SL = $50,000 × (1 - 0.02) = $49,000

✅ AI can set SL anywhere at or above $49,000
```

### Example 2: 20x Leverage
```
Entry: $3,000
Leverage: 20x
Max margin loss: 10%

allowed_move_pct = 10% / 20 = 0.5%
Risk floor SL = $3,000 × (1 - 0.005) = $2,985

⚠️ If AI sets SL at $2,950, system tightens it to $2,985
```

### Example 3: 50x Leverage (Over-Leveraged)
```
Entry: $100
Leverage: 50x
Max margin loss: 10%

allowed_move_pct = 10% / 50 = 0.2%

🔴 Cannot set realistic SL within risk budget
→ FULL_EXIT_NOW triggered immediately
```

---

## 🎛️ Configuration

**Location:** `backend/domains/exits/exit_brain_v3/dynamic_executor.py`

```python
# Max margin loss per trade (default 10%)
MAX_MARGIN_LOSS_PER_TRADE_PCT = 0.10

# Min practical stop distance (default 0.2%)
MIN_PRICE_STOP_DISTANCE_PCT = 0.002
```

**To adjust risk tolerance:**
- **Conservative:** `0.05` (5% max margin loss)
- **Moderate:** `0.10` (10% max margin loss) ← DEFAULT
- **Aggressive:** `0.15` (15% max margin loss)

---

## 🔍 Monitoring

```powershell
# Watch all risk management activity
docker logs quantum_backend --tail 100 --follow | Select-String "EXIT_BRAIN_RISK"

# Check for SL adjustments
docker logs quantum_backend --tail 200 | Select-String "tightening to final_sl"

# Check for over-leverage exits
docker logs quantum_backend --tail 200 | Select-String "over-leverage"
```

---

## 📊 Log Patterns

### Normal Operation
```
[EXIT_BRAIN_RISK] BTCUSDT LONG: entry=$50000.0000, leverage=5.0x, 
max_margin_loss=10.0%, allowed_move_pct=2.000%, risk_sl_price=$49000.0000
```

### SL Tightened
```
[EXIT_BRAIN_RISK] ETHUSDT LONG: strategic_sl=$2950.00 below risk_floor=$2985.00 
→ tightening to final_sl=$2985.00
```

### Over-Leverage Force Exit
```
[EXIT_BRAIN_RISK] SOLUSDT SHORT: allowed_move_pct=0.200% < 
MIN_PRICE_STOP_DISTANCE_PCT=0.200% 
→ FULL_EXIT_NOW triggered due to over-leverage vs risk budget
```

---

## 🎯 How It Works

```
1. Position Opened
   ↓
2. Extract Leverage from Binance
   ↓
3. Calculate Risk Floor SL
   allowed_move_pct = MAX_MARGIN_LOSS / leverage
   risk_sl = entry × (1 ± allowed_move_pct)
   ↓
4. Check Over-Leverage
   if allowed_move_pct < 0.2% → FORCE EXIT
   ↓
5. Apply Constraint
   LONG:  final_sl = max(ai_sl, risk_sl)
   SHORT: final_sl = min(ai_sl, risk_sl)
   ↓
6. Set Active SL
```

---

## ✅ What's Protected

**✅ Guaranteed:**
- No single trade can lose more than configured % of margin
- Works with ALL leverage levels (1x to 125x)
- Applies to ALL positions automatically

**✅ Compatibility:**
- Works with hybrid SL model
- Works with AI Exit Adapter (9C-16)
- Works with Exit Planner
- Does NOT break existing machinery

---

## 🚨 When It Triggers

### Risk Floor Tightening
**When:** AI sets SL that would allow >10% margin loss  
**Action:** Automatically tightens SL to risk floor  
**Impact:** Smaller potential loss, earlier exit possible

### Over-Leverage Force Exit
**When:** Leverage so high that min stop distance < 0.2%  
**Action:** Immediate FULL_EXIT_NOW at market  
**Impact:** Position closed before significant loss

### Initial SL Setting
**When:** Position opened but AI hasn't set SL yet  
**Action:** Risk floor applied as initial SL  
**Impact:** Position protected immediately

---

## 🔧 Troubleshooting

### Too Many Force Exits?
**Problem:** Over-leverage warnings too frequent  
**Solution:** Reduce position leverage OR lower MIN_PRICE_STOP_DISTANCE_PCT

### AI SL Always Overridden?
**Problem:** Risk floor constantly tightening AI SL  
**Solution:** Increase MAX_MARGIN_LOSS_PER_TRADE_PCT OR reduce leverage

### No Risk Logs?
**Problem:** EXIT_BRAIN_RISK messages not appearing  
**Check:** Exit Brain V3 active? Positions open? MOVE_SL decisions happening?

---

## 📈 Max Leverage by Risk Level

With default 10% max margin loss and 0.2% min stop distance:

| Leverage | Allowed Move | Status |
|----------|--------------|--------|
| 1x | 10.00% | ✅ Very Safe |
| 5x | 2.00% | ✅ Safe |
| 10x | 1.00% | ✅ Safe |
| 20x | 0.50% | ⚠️ Moderate |
| 30x | 0.33% | ⚠️ Moderate |
| 40x | 0.25% | ⚠️ Tight |
| 50x | 0.20% | 🔴 Critical |
| 51x+ | <0.20% | 🔴 Over-Leveraged → Force Exit |

**Recommendation:** Keep leverage ≤ 30x for comfortable SL placement

---

## 📞 Quick Help

**Problem:** Need to change risk tolerance  
**File:** `backend/domains/exits/exit_brain_v3/dynamic_executor.py`  
**Line:** ~57-58  
**Change:** `MAX_MARGIN_LOSS_PER_TRADE_PCT = 0.10` to desired value  
**Rebuild:** `docker compose --profile dev build backend`  
**Restart:** `docker compose --profile dev restart backend`

---

## 🎯 Deployment Status

**Date:** 2024  
**Status:** ✅ **LIVE IN PRODUCTION**  
**Mode:** EXIT BRAIN V3 LIVE MODE  
**Protection:** ACTIVE on all positions

---

**Full Documentation:** `AI_LEVERAGE_AWARE_RISK_MANAGEMENT.md`
