# Exit Brain V3: Dynamic TP Quick Reference

## 🎯 Quick Overview

**What's New:**
- ✅ Dynamic partial TP execution (fractions apply to remaining size)
- ✅ Automatic SL ratcheting (breakeven after TP1, TP1 price after TP2)
- ✅ Max unrealized loss guard (emergency exit at -12.5% PnL)

**Files Modified:**
- `types.py` - Extended PositionExitState
- `dynamic_executor.py` - Added loss guard + ratcheting logic
- `test_dynamic_executor_partial_tp.py` - 15+ unit tests

---

## 🔧 Configuration Quick Reference

```python
# In dynamic_executor.py (lines 56-66)

# Loss Guard
MAX_UNREALIZED_LOSS_PCT_PER_POSITION = 12.5  # Emergency exit threshold

# TP Configuration
DYNAMIC_TP_PROFILE_DEFAULT = [0.25, 0.25, 0.50]  # Fallback distribution
MAX_TP_LEVELS = 4  # Maximum TP levels

# Ratcheting
RATCHET_SL_ENABLED = True  # Toggle SL ratcheting
```

**To disable ratcheting:**
```python
RATCHET_SL_ENABLED = False
```

**To adjust loss guard threshold:**
```python
MAX_UNREALIZED_LOSS_PCT_PER_POSITION = 15.0  # Wider tolerance
```

---

## 📊 State Fields Reference

```python
@dataclass
class PositionExitState:
    # ... existing fields ...
    
    # NEW: Dynamic TP tracking
    entry_price: Optional[float] = None           # Entry price
    initial_size: Optional[float] = None          # Original size
    remaining_size: Optional[float] = None        # Current remaining
    tp_hits_count: int = 0                        # Number of TPs hit
    max_unrealized_profit_pct: float = 0.0        # High-water mark
    loss_guard_triggered: bool = False            # Prevent double-fire
```

**Auto-initialization:**
- `initial_size` = `position_size` on creation
- `remaining_size` = `position_size` on creation
- Updated automatically after each TP trigger

---

## 🔍 Log Patterns Cheat Sheet

### TP Trigger Execution
```
[EXIT_TP_TRIGGER] 🎯 SYMBOL SIDE: TP{N} HIT @ ${price} (TP=${tp_price}) - Closing {qty} ({pct}% of position) with MARKET {side}
[EXIT_ORDER] ✅ TP{N} MARKET {side} SYMBOL {qty} executed successfully - orderId={id}
[EXIT_TP_TRIGGER] 📊 SYMBOL SIDE: Updated state after TP{N} - remaining_size={size}, tp_hits_count={count}
```

### SL Ratcheting
```
[EXIT_RATCHET_SL] 🎯 SYMBOL SIDE: SL ratcheted ${old_sl} → ${new_sl} - {reason} (tp_hits={count})
```

**Ratchet reasons:**
- `breakeven after TP1` - Moved to entry price after first TP
- `TP1 price after TP2 hit` - Moved to first TP price after second TP

### Loss Guard
```
[EXIT_LOSS_GUARD] 🚨 SYMBOL SIDE: LOSS GUARD TRIGGERED - unrealized_pnl={pnl}% <= -{threshold}%
[EXIT_LOSS_GUARD] 🚨 SYMBOL SIDE: Closing FULL position {qty} with MARKET {side} (PnL={pnl}%)
[EXIT_ORDER] ✅ LOSS GUARD MARKET {side} SYMBOL {qty} executed - orderId={id}
```

### TP Fraction Normalization
```
[EXIT_BRAIN_STATE] SYMBOL SIDE: TP fractions sum={sum} > 1.0, normalized to 1.0
[EXIT_BRAIN_STATE] SYMBOL SIDE: TP fractions sum={sum} < 1.0, remaining {pct}% will be runner
```

---

## 🎬 Execution Flow

### Monitoring Cycle (Every 10s)
```
1. Fetch open positions
2. For each position:
   a. Build/update PositionExitState
   b. CHECK LOSS GUARD FIRST ← NEW (highest priority)
      → If triggered: close position, skip rest
   c. Get AI decision from adapter
   d. Update state from decision (SL/TP levels)
   e. Check for triggered levels (SL/TP)
3. Execute MARKET orders for triggered levels
```

### TP Trigger Flow
```
1. Price hits TP level
2. Calculate close_qty = remaining_size * size_pct ← Note: REMAINING, not initial
3. Submit MARKET reduce-only order
4. On success:
   a. Add to triggered_legs
   b. Update position_size
   c. Update remaining_size ← NEW
   d. Increment tp_hits_count ← NEW
   e. Call _recompute_dynamic_tp_and_sl() ← NEW
      → Ratchet SL based on tp_hits_count
```

### Loss Guard Check
```
1. Get unrealized_pnl_pct from PositionContext
2. If pnl <= -MAX_UNREALIZED_LOSS_PCT:
   a. Log trigger
   b. Close full remaining_size with MARKET
   c. Set loss_guard_triggered = True
   d. Clear state (sl/tp levels)
   e. Return True (skip further processing)
```

---

## 🧪 Testing Quick Start

### Run All Tests
```bash
pytest backend/domains/exits/exit_brain_v3/test_dynamic_executor_partial_tp.py -v
```

### Run Specific Test Class
```bash
# Test SL ratcheting only
pytest backend/domains/exits/exit_brain_v3/test_dynamic_executor_partial_tp.py::TestSLRatcheting -v

# Test loss guard only
pytest backend/domains/exits/exit_brain_v3/test_dynamic_executor_partial_tp.py::TestLossGuard -v
```

### Test with Coverage
```bash
pytest backend/domains/exits/exit_brain_v3/test_dynamic_executor_partial_tp.py \
  --cov=backend.domains.exits.exit_brain_v3.dynamic_executor \
  --cov-report=html
```

---

## 🐛 Debugging Tips

### Check Loss Guard Status
```bash
docker logs quantum_backend 2>&1 | grep "EXIT_LOSS_GUARD"
```

### Check SL Ratcheting
```bash
docker logs quantum_backend 2>&1 | grep "EXIT_RATCHET_SL"
```

### Check TP Execution with State Updates
```bash
docker logs quantum_backend 2>&1 | grep -E "EXIT_TP_TRIGGER|remaining_size|tp_hits_count"
```

### Monitor All Dynamic TP Activity
```bash
docker logs -f quantum_backend 2>&1 | grep -E "EXIT_RATCHET_SL|EXIT_LOSS_GUARD|EXIT_TP_TRIGGER"
```

### Check State for Specific Symbol
```bash
docker logs quantum_backend 2>&1 | grep "BTCUSDT" | grep -E "EXIT_TP_TRIGGER|EXIT_RATCHET_SL|remaining_size"
```

---

## 🔧 Common Adjustments

### Disable Ratcheting Temporarily
```python
# In dynamic_executor.py
RATCHET_SL_ENABLED = False  # Line 64
```

### Widen Loss Guard Threshold
```python
# In dynamic_executor.py
MAX_UNREALIZED_LOSS_PCT_PER_POSITION = 15.0  # Was 12.5 (line 59)
```

### Adjust TP Level Cap
```python
# In dynamic_executor.py
MAX_TP_LEVELS = 6  # Was 4 (line 66)
```

### Change Default TP Distribution
```python
# In dynamic_executor.py
DYNAMIC_TP_PROFILE_DEFAULT = [0.30, 0.30, 0.40]  # Was [0.25, 0.25, 0.50] (line 62)
```

---

## 📈 Example Scenarios

### Scenario 1: BTCUSDT LONG with Full TP Ladder

**Position:**
```
Symbol: BTCUSDT
Side: LONG
Entry: $50,000
Size: 1.0 BTC
Leverage: 20x
```

**TP Levels:**
```
TP1: $51,500 (25%) - +3.0% price, +60% PnL @ 20x
TP2: $52,500 (25%) - +5.0% price, +100% PnL @ 20x
TP3: $54,000 (50%) - +8.0% price, +160% PnL @ 20x
```

**Initial SL:**
```
SL: $49,000 (-2.0% price, -40% PnL @ 20x)
```

**Execution Timeline:**

**T0 - Position Open:**
```
remaining_size: 1.0 BTC
tp_hits_count: 0
active_sl: $49,000
```

**T1 - TP1 Triggers @ $51,500:**
```
[EXIT_TP_TRIGGER] 🎯 BTCUSDT LONG: TP1 HIT @ $51500.00 - Closing 0.25 (25% of position)
→ remaining_size: 0.75 BTC
→ tp_hits_count: 1
[EXIT_RATCHET_SL] 🎯 BTCUSDT LONG: SL ratcheted $49000 → $50000 - breakeven after TP1 (tp_hits=1)
→ active_sl: $50,000 (locked in breakeven)
```

**T2 - TP2 Triggers @ $52,500:**
```
[EXIT_TP_TRIGGER] 🎯 BTCUSDT LONG: TP2 HIT @ $52500.00 - Closing 0.1875 (25% of remaining 0.75)
→ remaining_size: 0.5625 BTC
→ tp_hits_count: 2
[EXIT_RATCHET_SL] 🎯 BTCUSDT LONG: SL ratcheted $50000 → $51500 - TP1 price after TP2 hit (tp_hits=2)
→ active_sl: $51,500 (locked in TP1 profit)
```

**T3 - TP3 Triggers @ $54,000:**
```
[EXIT_TP_TRIGGER] 🎯 BTCUSDT LONG: TP3 HIT @ $54000.00 - Closing 0.28125 (50% of remaining 0.5625)
→ remaining_size: 0.28125 BTC (runner)
→ tp_hits_count: 3
→ active_sl: $51,500 (unchanged, no more ratcheting rules)
```

**Final State:**
```
Closed: 0.71875 BTC (71.875% of original)
Remaining: 0.28125 BTC (28.125% runner)
SL Protection: $51,500 (+3% from entry, profit locked)
```

---

### Scenario 2: Loss Guard Trigger

**Position:**
```
Symbol: ETHUSDT
Side: LONG
Entry: $3,000
Size: 10 ETH
Leverage: 20x
unrealized_pnl: -12.6% (exceeds -12.5% threshold)
```

**Execution:**
```
[EXIT_LOSS_GUARD] 🚨 ETHUSDT LONG: LOSS GUARD TRIGGERED - unrealized_pnl=-12.60% <= -12.50%
[EXIT_LOSS_GUARD] 🚨 ETHUSDT LONG: Closing FULL position 10.0 with MARKET SELL (PnL=-12.60%)
[EXIT_ORDER] ✅ LOSS GUARD MARKET SELL ETHUSDT 10.0 executed - orderId=789012

State cleared:
→ remaining_size: 0
→ loss_guard_triggered: True
→ active_sl: None
→ tp_levels: []
```

---

## 🚨 Troubleshooting

### Issue: SL Not Ratcheting After TP Hit

**Check:**
```python
# 1. Is ratcheting enabled?
RATCHET_SL_ENABLED = True  # Must be True

# 2. Verify tp_hits_count incremented
# Look for log: "Updated state after TP{N} - ... tp_hits_count={count}"

# 3. Verify entry_price is set
# Check state creation log: "Created new state for {key} - entry=${price}"
```

### Issue: Loss Guard Not Triggering

**Check:**
```python
# 1. Verify unrealized_pnl_pct format
# Must be percentage (e.g., -12.6, not -0.126)

# 2. Check threshold
MAX_UNREALIZED_LOSS_PCT_PER_POSITION = 12.5  # Default

# 3. Not already triggered?
# loss_guard_triggered must be False
```

### Issue: TP Closing Wrong Quantity

**Check:**
```python
# 1. Verify remaining_size is updated
# Should decrease after each TP execution

# 2. Check quantization
# Look for log: "close_qty={qty} invalid after quantization"

# 3. Verify size_pct in tp_levels
# Fractions should sum to ≤ 1.0
```

---

## 📞 Support

**For questions:**
- Check logs with patterns above
- Review unit tests for expected behavior
- See full documentation: `AI_EXIT_BRAIN_DYNAMIC_TP_IMPLEMENTATION.md`

**Report issues:**
- Include symbol, side, and relevant log snippets
- State configuration (RATCHET_SL_ENABLED, MAX_UNREALIZED_LOSS_PCT, etc.)
- Expected vs actual behavior

---

**Last Updated:** December 11, 2025  
**Version:** Exit Brain V3 + Dynamic TP v1.0
