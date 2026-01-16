# ORCHESTRATOR PROFILES: SAFE vs AGGRESSIVE

## 📋 OVERVIEW

Two complete orchestrator profiles have been implemented to support different trading scenarios:

1. **SAFE Profile** - For real capital and low drawdown tolerance
2. **AGGRESSIVE Profile** - For testnet and faster experimentation

---

## 🎯 PROFILE COMPARISON

### Quick Reference Table

| Parameter | SAFE Profile | AGGRESSIVE Profile | Description |
|-----------|--------------|-------------------|-------------|
| **Base Confidence** | 0.55 (55%) | 0.45 (45%) | Entry signal threshold |
| **Base Risk %** | 0.8% | 1.2% | Risk per trade |
| **Daily DD Limit** | 2.5% | 4.5% | Max daily drawdown before stop |
| **Losing Streak Limit** | 4 losses | 7 losses | Consecutive losses before pause |
| **Max Open Positions** | 5 | 10 | Simultaneous trades allowed |
| **Total Exposure Limit** | 10% | 20% | Max portfolio exposure |
| **Extreme Vol Threshold** | 0.05 (5%) | 0.08 (8%) | ATR/price ratio trigger |
| **High Vol Threshold** | 0.03 (3%) | 0.05 (5%) | Volatility sensitivity |
| **High Spread BPS** | 8.0 | 15.0 | Spread cost tolerance |
| **High Slippage BPS** | 6.0 | 12.0 | Slippage cost tolerance |

---

## 🛡️ SAFE PROFILE CHARACTERISTICS

### Philosophy
**Conservative, Capital Preservation, Lower Drawdown**

### Risk Scaling Multipliers

```python
"BULL": 0.9          # Slightly conservative even in bull markets
"BEAR": 0.3          # Very defensive in bear markets  
"HIGH_VOL": 0.4      # Aggressive risk cut in high volatility
"CHOP": 0.5          # Reduce risk in choppy conditions
"NORMAL": 0.8        # Default conservative multiplier
```

### Confidence Adjustments

```python
"BULL": 0.00         # No increase
"BEAR": +0.08        # Require 63% confidence (55% + 8%)
"HIGH_VOL": +0.10    # Require 65% confidence (55% + 10%)
"CHOP": +0.05        # Require 60% confidence
```

### Symbol Performance Thresholds

```python
"min_winrate": 0.45          # Block symbols with <45% win rate
"min_avg_R": 0.6             # Block symbols with <0.6 R-multiple
"bad_streak_limit": 3        # Block after 3 consecutive losses
```

### Exit Mode Bias

```python
"BULL": "TREND_FOLLOW"       # Follow trends in bull
"BEAR": "FAST_TP"            # Take profits quickly
"HIGH_VOL": "DEFENSIVE_TRAIL" # Tight stops
"CHOP": "FAST_TP"            # Quick exits
```

### Recovery Behavior

```python
"recovery_multiplier": 1.1    # Slow recovery: +10% risk after win
"recovery_after_streak": 2    # Need 2 wins to fully recover
```

### Cost Sensitivity

```python
"cost_sensitivity": "HIGH"   # React strongly to costs
"max_cost_in_R": 0.15         # Block if costs > 0.15R
```

---

## ⚡ AGGRESSIVE PROFILE CHARACTERISTICS

### Philosophy
**Growth-Oriented, Higher Risk Tolerance, Faster Experimentation**

### Risk Scaling Multipliers

```python
"BULL": 1.3          # Capitalize on bull runs (+30%)
"BEAR": 0.6          # Still trade in bears (reduced)
"HIGH_VOL": 0.8      # Less reduction in volatility
"CHOP": 0.7          # More active in chop
"NORMAL": 1.0        # Full risk in normal conditions
```

### Confidence Adjustments

```python
"BULL": -0.02        # LOWER threshold to 43% (more trades!)
"BEAR": +0.05        # Only 50% required (vs 53% in SAFE)
"HIGH_VOL": +0.05    # Only 50% required
"CHOP": +0.02        # 47% required
```

### Symbol Performance Thresholds

```python
"min_winrate": 0.35          # Allow symbols with 35%+ win rate
"min_avg_R": 0.3             # Allow lower R-multiples
"bad_streak_limit": 5        # Tolerate 5 losses before blocking
```

### Exit Mode Bias

```python
"BULL": "TREND_FOLLOW"       # Maximize bull moves
"BEAR": "TREND_FOLLOW"       # Still follow trends (contrarian)
"HIGH_VOL": "TREND_FOLLOW"   # Ride volatility
"CHOP": "TREND_FOLLOW"       # Try to catch breakouts
```

### Recovery Behavior

```python
"recovery_multiplier": 1.3    # Fast recovery: +30% risk after win
"recovery_after_streak": 1    # Just 1 win needed to recover
```

### Cost Sensitivity

```python
"cost_sensitivity": "LOW"    # Less concerned about costs
"max_cost_in_R": 0.30         # Allow up to 0.30R in costs
```

---

## 🔧 USAGE EXAMPLES

### Method 1: Environment Variable (Recommended)

```bash
# Windows PowerShell
$env:ORCH_PROFILE="SAFE"
systemctl restart backend

# Linux/Mac
export ORCH_PROFILE=SAFE
systemctl restart backend

# For AGGRESSIVE profile
$env:ORCH_PROFILE="AGGRESSIVE"
```

### Method 2: Direct Code Usage

```python
from backend.services.orchestrator_policy import OrchestratorPolicy

# Use active profile from environment (default: SAFE)
orchestrator = OrchestratorPolicy()

# Explicitly use SAFE profile
orchestrator = OrchestratorPolicy(profile_name="SAFE")

# Explicitly use AGGRESSIVE profile
orchestrator = OrchestratorPolicy(profile_name="AGGRESSIVE")

# Custom config (ignores profiles)
from backend.services.orchestrator_policy import OrchestratorConfig
custom_config = OrchestratorConfig(
    base_confidence=0.60,
    base_risk_pct=0.9
)
orchestrator = OrchestratorPolicy(config=custom_config)
```

### Method 3: Load Profile Directly

```python
from backend.services.orchestrator_config import load_profile, get_active_profile

# Load specific profile
safe_profile = load_profile("SAFE")
print(safe_profile["base_confidence"])  # 0.55

# Get currently active profile
active_profile = get_active_profile()
print(active_profile["base_risk_pct"])  # Depends on ORCH_PROFILE env
```

---

## 📊 EXAMPLE SCENARIOS

### Scenario 1: SAFE Profile in BULL Market

```
Conditions: BULL regime, normal volatility, no drawdown
Base Risk: 0.8%
Regime Multiplier: 0.9 (BULL)
Confidence Adjustment: 0.00 (no increase)

→ Actual Risk: 0.8% × 0.9 = 0.72%
→ Min Confidence: 55% + 0% = 55%
→ Exit Mode: TREND_FOLLOW
→ Entry Mode: NORMAL
```

### Scenario 2: SAFE Profile in HIGH_VOL

```
Conditions: HIGH_VOL regime, 3 consecutive losses, -1.5% DD
Base Risk: 0.8%
Regime Multiplier: 0.4 (HIGH_VOL)
Losing Streak: 3 × 0.15 = 0.45 reduction → ×0.55
Drawdown: 1.5 × 0.10 = 0.15 reduction → ×0.85
Confidence Adjustment: +0.10 (require 65%)

→ Actual Risk: 0.8% × 0.4 × 0.55 × 0.85 = 0.15%
→ Min Confidence: 55% + 10% = 65%
→ Exit Mode: DEFENSIVE_TRAIL
→ Entry Mode: DEFENSIVE
```

### Scenario 3: AGGRESSIVE Profile in BULL Market

```
Conditions: BULL regime, normal volatility, no drawdown
Base Risk: 1.2%
Regime Multiplier: 1.3 (BULL)
Confidence Adjustment: -0.02 (LOWER threshold!)

→ Actual Risk: 1.2% × 1.3 = 1.56%
→ Min Confidence: 45% - 2% = 43%
→ Exit Mode: TREND_FOLLOW
→ Entry Mode: AGGRESSIVE
→ Max Positions: 10
```

### Scenario 4: AGGRESSIVE Profile in BEAR

```
Conditions: BEAR regime, normal volatility, no drawdown
Base Risk: 1.2%
Regime Multiplier: 0.6 (BEAR)
Confidence Adjustment: +0.05

→ Actual Risk: 1.2% × 0.6 = 0.72%
→ Min Confidence: 45% + 5% = 50%
→ Exit Mode: TREND_FOLLOW (still trend-following)
→ Entry Mode: NORMAL
```

---

## 🎯 WHEN TO USE EACH PROFILE

### Use SAFE Profile When:

- ✅ Trading with **real capital**
- ✅ Low **drawdown tolerance** (<3%)
- ✅ Need **capital preservation**
- ✅ Volatile market conditions
- ✅ Testing new strategies conservatively
- ✅ Small account size (need to protect every trade)
- ✅ High **risk aversion**

### Use AGGRESSIVE Profile When:

- ✅ Trading on **testnet** (paper trading)
- ✅ Experimenting with new features
- ✅ Want **faster data collection**
- ✅ Comfortable with 4-5% drawdown
- ✅ Larger account size (can absorb losses)
- ✅ Strong bull market conditions
- ✅ Testing maximum system capacity

---

## 🔍 PROFILE BEHAVIOR MATRIX

### Position Size Example (10k Account)

| Scenario | SAFE Risk | SAFE Position | AGGRESSIVE Risk | AGGRESSIVE Position |
|----------|-----------|---------------|-----------------|---------------------|
| **BULL + Normal** | 0.72% | $72 | 1.56% | $156 |
| **NORMAL + 0 Losses** | 0.80% | $80 | 1.20% | $120 |
| **HIGH_VOL + Normal** | 0.32% | $32 | 0.96% | $96 |
| **BEAR + Normal** | 0.24% | $24 | 0.72% | $72 |
| **CHOP + Normal** | 0.40% | $40 | 0.84% | $84 |
| **BULL + 3 Losses** | 0.28% | $28 | 1.00% | $100 |

### Confidence Thresholds

| Regime | SAFE Min Conf | AGGRESSIVE Min Conf | Difference |
|--------|---------------|---------------------|------------|
| **BULL** | 55% | 43% | 12% more trades for AGG |
| **NORMAL** | 55% | 45% | 10% more trades for AGG |
| **HIGH_VOL** | 65% | 50% | 15% more trades for AGG |
| **BEAR** | 63% | 50% | 13% more trades for AGG |
| **CHOP** | 60% | 47% | 13% more trades for AGG |

### Symbol Tolerance

| Metric | SAFE Threshold | AGGRESSIVE Threshold |
|--------|---------------|---------------------|
| **Min Win Rate** | 45% | 35% |
| **Min Avg R** | 0.6 | 0.3 |
| **Bad Streak Limit** | 3 losses | 5 losses |

---

## 🚨 IMPORTANT NOTES

### Profile Loading Priority

1. **Explicit `config` parameter** (highest priority)
2. **Explicit `profile_name` parameter**
3. **`ORCH_PROFILE` environment variable**
4. **Default: SAFE profile** (lowest priority)

### Switching Profiles

**Requires backend restart:**

```bash
# Change profile
$env:ORCH_PROFILE="AGGRESSIVE"

# Restart backend to apply
systemctl restart backend

# Verify active profile
journalctl -u quantum_backend.service | Select-String "Loading.*profile"
```

### Profile Validation

```bash
# Check current profile
python check_profile_status.py
```

### Profile Best Practices

1. **Start with SAFE** on real capital
2. **Use AGGRESSIVE** only on testnet initially
3. **Monitor drawdown** closely with AGGRESSIVE
4. **Switch to SAFE** if DD approaches 4%
5. **Test profile changes** on testnet first

---

## 📝 VERIFICATION

### Check Active Profile

```python
from backend.services.orchestrator_config import CURRENT_PROFILE, get_active_profile

print(f"Active Profile: {CURRENT_PROFILE}")
profile = get_active_profile()
print(f"Base Confidence: {profile['base_confidence']}")
print(f"Base Risk: {profile['base_risk_pct']}%")
```

### Logs to Monitor

```bash
# Check profile loading
journalctl -u quantum_backend.service | Select-String "Loading.*profile"

# Example output:
# 🛡️ Loading SAFE profile: Conservative risk, higher thresholds
# 🎯 Using profile: SAFE
# ✅ OrchestratorPolicy initialized: Base confidence=0.55, Base risk=0.80%
```

---

## 🎓 SUMMARY

### SAFE Profile = **Defensive**
- Higher confidence (55%)
- Lower risk (0.8%)
- Stricter limits (2.5% DD, 4 losses)
- Fast profit-taking
- Symbol quality required (45% WR, 0.6 R)
- Slow recovery

### AGGRESSIVE Profile = **Offensive**
- Lower confidence (45%)
- Higher risk (1.2%)
- Relaxed limits (4.5% DD, 7 losses)
- Trend-following everywhere
- Symbol quality tolerant (35% WR, 0.3 R)
- Fast recovery

**Use the right tool for the right job!** 🛡️⚡

