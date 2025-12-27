# RL VOLATILITY SAFETY ENVELOPE - QUICK REFERENCE 🛡️

Volatility-based safety layer for RL position sizing - Quick lookup for developers

---

## 🚀 QUICK START

```python
from backend.services.risk.rl_volatility_safety_envelope import get_rl_volatility_envelope

# Initialize envelope
envelope = get_rl_volatility_envelope(policy_store)

# After RL decision
result = envelope.apply_limits(
    symbol="BTCUSDT",
    atr_pct=0.025,  # 2.5% ATR
    proposed_leverage=20.0,
    proposed_risk_pct=0.08,
    equity_usd=10000.0
)

# Use capped values
final_leverage = result.capped_leverage
final_position_size = result.capped_risk_pct * equity_usd
```

---

## 📊 VOLATILITY BUCKETS

| Bucket      | ATR/Price | Max Leverage | Max Risk % |
|-------------|-----------|--------------|------------|
| **LOW**     | < 0.5%    | 25x          | 10%        |
| **NORMAL**  | 0.5-1.5%  | 20x          | 8%         |
| **HIGH**    | 1.5-3.0%  | 15x          | 5%         |
| **EXTREME** | > 3.0%    | 10x          | 3%         |

---

## ⚙️ POLICYSTORE KEYS

```python
# Default configuration:
volatility.low.max_leverage = 25.0
volatility.low.max_risk_pct = 0.10

volatility.normal.max_leverage = 20.0
volatility.normal.max_risk_pct = 0.08

volatility.high.max_leverage = 15.0
volatility.high.max_risk_pct = 0.05

volatility.extreme.max_leverage = 10.0
volatility.extreme.max_risk_pct = 0.03
```

---

## 🔧 COMMON OPERATIONS

### Check Volatility Bucket
```python
atr_pct = atr / price  # e.g., 0.025 = 2.5%
bucket = envelope.get_volatility_bucket(atr_pct)
print(bucket)  # VolatilityBucket.HIGH
```

### Apply Limits
```python
result = envelope.apply_limits(
    symbol="ETHUSDT",
    atr_pct=0.025,
    proposed_leverage=20.0,
    proposed_risk_pct=0.08,
    equity_usd=10000.0
)

print(result.was_capped)  # True or False
print(result.capped_leverage)  # 15.0 (capped in HIGH vol)
print(result.capped_risk_pct)  # 0.05 (capped in HIGH vol)
```

### Calculate Position Size
```python
margin, quantity = envelope.calculate_capped_position_size(
    equity_usd=10000.0,
    capped_risk_pct=0.05,
    capped_leverage=15.0,
    price=50000.0
)
# margin = $500, quantity = 0.15 BTC
```

### Get Status
```python
status = envelope.get_status()
print(status['thresholds'])  # ATR thresholds
print(status['cached_limits'])  # Cached bucket limits
```

---

## 🔍 INTEGRATION FLOW

```
1. RL Agent decides: leverage=20x, size=$800
         ↓
2. Envelope checks ATR: 2.5% = HIGH volatility
         ↓
3. Envelope caps: leverage=15x, size=$500
         ↓
4. Safety Governor validates capped values
         ↓
5. Order submitted with safe parameters
```

---

## 📈 EXAMPLES

### Example 1: Normal Volatility - No Capping
```python
# Market: 1.0% ATR (NORMAL)
# RL: 18x leverage, $700 position
result = envelope.apply_limits(
    symbol="BTCUSDT",
    atr_pct=0.01,
    proposed_leverage=18.0,
    proposed_risk_pct=0.07,
    equity_usd=10000.0
)
# Result: 18x, $700 (unchanged - within NORMAL limits)
```

### Example 2: High Volatility - Capping
```python
# Market: 2.5% ATR (HIGH)
# RL: 20x leverage, $800 position
result = envelope.apply_limits(
    symbol="ETHUSDT",
    atr_pct=0.025,
    proposed_leverage=20.0,
    proposed_risk_pct=0.08,
    equity_usd=10000.0
)
# Result: 15x, $500 (CAPPED to HIGH limits)
```

### Example 3: Extreme Volatility - Heavy Capping
```python
# Market: 4.0% ATR (EXTREME)
# RL: 25x leverage, $1000 position
result = envelope.apply_limits(
    symbol="SOLUSDT",
    atr_pct=0.04,
    proposed_leverage=25.0,
    proposed_risk_pct=0.10,
    equity_usd=10000.0
)
# Result: 10x, $300 (HEAVILY CAPPED to EXTREME limits)
```

---

## 🚨 TROUBLESHOOTING

### Envelope Not Capping

**Check:**
1. ATR data available: `market_data['atr']`
2. Envelope initialized: `self.rl_envelope is not None`
3. PolicyStore limits: `policy_store.get("volatility.high.max_leverage")`

### Unexpected Capping

**Check:**
1. Current ATR: `atr_pct = atr / price`
2. Volatility bucket: `envelope.get_volatility_bucket(atr_pct)`
3. PolicyStore limits: `envelope.get_limits_for_bucket(bucket)`

### Envelope Failing

**Check logs:**
```bash
grep "RL-ENVELOPE" logs/app.log
```

**Expected:**
```
[OK] RL Volatility Safety Envelope available
[OK] RL Volatility Safety Envelope initialized
🛡️ [RL-ENVELOPE] BTCUSDT | HIGH volatility | ...
```

---

## 📝 CONFIGURATION RECIPES

### Conservative (Testnet)
```python
policy_store.set("volatility.normal.max_leverage", 15.0)
policy_store.set("volatility.normal.max_risk_pct", 0.05)
policy_store.set("volatility.high.max_leverage", 10.0)
policy_store.set("volatility.high.max_risk_pct", 0.03)
```

### Balanced (Production)
```python
policy_store.set("volatility.normal.max_leverage", 20.0)
policy_store.set("volatility.normal.max_risk_pct", 0.08)
policy_store.set("volatility.high.max_leverage", 15.0)
policy_store.set("volatility.high.max_risk_pct", 0.05)
```

### Aggressive (High Risk)
```python
policy_store.set("volatility.normal.max_leverage", 25.0)
policy_store.set("volatility.normal.max_risk_pct", 0.10)
policy_store.set("volatility.high.max_leverage", 20.0)
policy_store.set("volatility.high.max_risk_pct", 0.08)
```

---

## 🎯 USE CASES

### Use Case 1: Flash Crash Protection
**Scenario:** BTC drops 20% in 1 hour  
**ATR:** 5.0% (EXTREME)  
**Without Envelope:** RL proposes 25x, $1,000  
**With Envelope:** Capped to 10x, $300  
**Result:** 88% risk reduction 🛡️

### Use Case 2: Normal Market
**Scenario:** Stable trending market  
**ATR:** 1.0% (NORMAL)  
**Without Envelope:** RL proposes 18x, $700  
**With Envelope:** No capping (within limits)  
**Result:** Full RL flexibility ✅

### Use Case 3: High Volatility Trading
**Scenario:** Elevated volatility period  
**ATR:** 2.5% (HIGH)  
**Without Envelope:** RL proposes 20x, $800  
**With Envelope:** Capped to 15x, $500  
**Result:** 53% risk reduction 🛡️

---

## 📞 SUPPORT

**Documentation:** `SPRINT1_D4_RL_VOLATILITY_ENVELOPE_COMPLETE.md`  
**Tests:** `tests/unit/test_rl_volatility_safety_envelope_sprint1_d4.py`  
**Source:** `backend/services/risk/rl_volatility_safety_envelope.py`  
**Integration:** `backend/services/execution/event_driven_executor.py` (lines ~1968-2009)

---

*Quick Reference v1.0 - December 4, 2025*
