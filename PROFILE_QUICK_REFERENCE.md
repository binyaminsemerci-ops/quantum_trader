# 🎯 PROFILE QUICK REFERENCE CARD

## 📋 AT A GLANCE

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR PROFILES                            │
│                                                                     │
│  🛡️  SAFE Profile              vs        ⚡ AGGRESSIVE Profile      │
│  ────────────────────                   ─────────────────────────  │
│  For: REAL CAPITAL                      For: TESTNET                │
│  Goal: Capital Preservation             Goal: Growth & Experimentation │
│  Risk: Conservative (0.8%)              Risk: High (1.2%)           │
│  Confidence: High (55%)                 Confidence: Low (45%)       │
│  Drawdown: 2.5% max                     Drawdown: 4.5% max          │
│  Positions: 5 max                       Positions: 10 max           │
│  Trading: Selective                     Trading: Active             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## ⚖️ RISK MULTIPLIERS BY REGIME

```
Market Condition    │  SAFE Profile  │  AGGRESSIVE Profile  │  Ratio
────────────────────┼────────────────┼──────────────────────┼────────
🐂 BULL Market      │     0.90x      │       1.30x          │  1.44x
🐻 BEAR Market      │     0.30x      │       0.60x          │  2.00x
📊 HIGH VOLATILITY  │     0.40x      │       0.80x          │  2.00x
🌊 CHOPPY           │     0.50x      │       0.70x          │  1.40x
➡️  NORMAL          │     0.80x      │       1.00x          │  1.25x
```

**Interpretation:**
- In BULL: AGGRESSIVE risks 1.44x more than SAFE (0.72% vs 1.56%)
- In BEAR: AGGRESSIVE risks 2x more than SAFE (0.24% vs 0.72%)
- In HIGH_VOL: AGGRESSIVE risks 2x more than SAFE (0.32% vs 0.96%)

---

## 🎲 CONFIDENCE THRESHOLDS BY REGIME

```
Market Condition    │  SAFE Min Conf  │  AGGRESSIVE Min Conf  │  Winner
────────────────────┼─────────────────┼───────────────────────┼─────────────
🐂 BULL Market      │      55%        │        43%            │  AGG (+12%)
🐻 BEAR Market      │      63%        │        50%            │  AGG (+13%)
📊 HIGH VOLATILITY  │      65%        │        50%            │  AGG (+15%)
🌊 CHOPPY           │      60%        │        47%            │  AGG (+13%)
➡️  NORMAL          │      55%        │        45%            │  AGG (+10%)
```

**Interpretation:**
- AGGRESSIVE allows trades at 10-15% lower confidence
- Translates to ~20-30% more trades executed
- SAFE requires higher certainty before entering

---

## 📈 POSITION SIZE EXAMPLES ($10,000 Account)

### Scenario 1: BULL Market, No Losses, Normal Vol

```
SAFE Profile:
  Base Risk: 0.8%
  BULL Multiplier: 0.9x
  → Actual Risk: 0.72%
  → Position Size: $72

AGGRESSIVE Profile:
  Base Risk: 1.2%
  BULL Multiplier: 1.3x
  → Actual Risk: 1.56%
  → Position Size: $156

💡 AGGRESSIVE risks 2.17x more per trade
```

### Scenario 2: HIGH_VOL, 3 Consecutive Losses, -1.5% DD

```
SAFE Profile:
  Base Risk: 0.8%
  HIGH_VOL Multiplier: 0.4x
  Losing Streak Penalty: ×0.55 (3 losses × 0.15)
  Drawdown Penalty: ×0.85 (1.5% × 0.10)
  → Actual Risk: 0.15%
  → Position Size: $15
  → Min Confidence: 65%

AGGRESSIVE Profile:
  Base Risk: 1.2%
  HIGH_VOL Multiplier: 0.8x
  Losing Streak Penalty: ×0.76 (3 losses × 0.08)
  Drawdown Penalty: ×0.925 (1.5% × 0.05)
  → Actual Risk: 0.68%
  → Position Size: $68
  → Min Confidence: 50%

💡 In adversity: AGGRESSIVE still risks 4.5x more than SAFE
```

### Scenario 3: BEAR Market, No Losses, Normal Vol

```
SAFE Profile:
  Base Risk: 0.8%
  BEAR Multiplier: 0.3x
  → Actual Risk: 0.24%
  → Position Size: $24
  → Min Confidence: 63%
  → Exit Mode: FAST_TP (take profits quickly)

AGGRESSIVE Profile:
  Base Risk: 1.2%
  BEAR Multiplier: 0.6x
  → Actual Risk: 0.72%
  → Position Size: $72
  → Min Confidence: 50%
  → Exit Mode: TREND_FOLLOW (still trend-following)

💡 In BEAR: AGGRESSIVE risks 3x more and trends instead of quick exits
```

---

## 🚪 EXIT STRATEGY BY REGIME

```
Market Condition    │  SAFE Exit Mode      │  AGGRESSIVE Exit Mode
────────────────────┼──────────────────────┼───────────────────────
🐂 BULL Market      │  TREND_FOLLOW        │  TREND_FOLLOW
🐻 BEAR Market      │  FAST_TP ⚡          │  TREND_FOLLOW 🎯
📊 HIGH VOLATILITY  │  DEFENSIVE_TRAIL 🛡️  │  TREND_FOLLOW 🎯
🌊 CHOPPY           │  FAST_TP ⚡          │  TREND_FOLLOW 🎯
```

**Key Difference:**
- **SAFE**: Switches to defensive exits in adverse conditions
- **AGGRESSIVE**: Always tries to follow trends for bigger gains

---

## 📊 SYMBOL QUALITY REQUIREMENTS

```
Metric                │  SAFE         │  AGGRESSIVE    │  Impact
──────────────────────┼───────────────┼────────────────┼─────────────────
Min Win Rate          │    45%        │     35%        │  AGG trades more symbols
Min Avg R-Multiple    │    0.6        │     0.3        │  AGG tolerates lower R
Bad Streak Limit      │  3 losses     │   5 losses     │  AGG more forgiving
```

**Example:**
- Symbol with 40% WR, 0.5 R-multiple, 4-loss streak:
  - ❌ BLOCKED by SAFE (below 45% WR)
  - ✅ ALLOWED by AGGRESSIVE (above 35% WR)

---

## 🔄 RECOVERY BEHAVIOR AFTER LOSSES

```
Profile      │  Recovery Mult  │  Wins Needed  │  After 1 Win  │  After 2 Wins
─────────────┼─────────────────┼───────────────┼───────────────┼────────────────
SAFE         │      1.1x       │       2       │   Not full    │   Full (100%)
AGGRESSIVE   │      1.3x       │       1       │   Full (100%) │   Full + growth
```

**Example: $10k Account, Currently at 0.5% Risk (Reduced from Loss)**

After 1 winning trade:
- SAFE: 0.5% × 1.1 = 0.55% (still not back to 0.8% base)
- AGGRESSIVE: 0.5% × 1.3 = 0.65% (closer to 1.2% base)

After 2 winning trades:
- SAFE: 0.55% × 1.1 = 0.605% → back to 0.8% (capped at base)
- AGGRESSIVE: 0.65% × 1.3 = 0.845% → can exceed base risk

💡 AGGRESSIVE recovers faster and can compound wins

---

## 💸 COST TOLERANCE

```
Cost Type            │  SAFE        │  AGGRESSIVE   │  Impact
─────────────────────┼──────────────┼───────────────┼──────────────────
Sensitivity Level    │  HIGH        │  LOW          │  SAFE more cautious
Max Spread (BPS)     │  8.0         │  15.0         │  AGG trades wider spreads
Max Slippage (BPS)   │  6.0         │  12.0         │  AGG tolerates more slippage
Max Cost in R        │  0.15R       │  0.30R        │  AGG accepts 2x costs
Confidence Penalty   │  +3% each    │  +1% each     │  SAFE penalizes more
```

**Example: Trade with 10 BPS Spread + 8 BPS Slippage**

SAFE:
- Spread: 10 BPS > 8 BPS threshold → +3% confidence penalty
- Slippage: 8 BPS > 6 BPS threshold → +3% confidence penalty
- Total: Min confidence rises from 55% to 61%
- Action: May block trade if signal confidence is 55-60%

AGGRESSIVE:
- Spread: 10 BPS < 15 BPS threshold → +1% confidence penalty
- Slippage: 8 BPS < 12 BPS threshold → +1% confidence penalty
- Total: Min confidence rises from 45% to 47%
- Action: Still allows trade

---

## 📉 DRAWDOWN BEHAVIOR

```
Drawdown Level    │  SAFE Risk Reduction  │  AGGRESSIVE Risk Reduction
──────────────────┼───────────────────────┼────────────────────────────
-0.5% DD          │    -5% (0.76% risk)   │    -2.5% (1.17% risk)
-1.0% DD          │   -10% (0.72% risk)   │    -5.0% (1.14% risk)
-1.5% DD          │   -15% (0.68% risk)   │    -7.5% (1.11% risk)
-2.0% DD          │   -20% (0.64% risk)   │   -10.0% (1.08% risk)
-2.5% DD          │  ⛔ TRADING PAUSED    │   -12.5% (1.05% risk)
-3.0% DD          │  ⛔ TRADING PAUSED    │   -15.0% (1.02% risk)
-4.5% DD          │  ⛔ TRADING PAUSED    │  ⛔ TRADING PAUSED
```

**Penalty Rates:**
- SAFE: 10% per 1% DD (aggressive reduction)
- AGGRESSIVE: 5% per 1% DD (slower reduction)

---

## 🎯 DECISION MATRIX: WHICH PROFILE?

### Choose SAFE if:

✅ Trading with **real money**  
✅ Cannot afford >2.5% drawdown  
✅ Want **capital preservation** over growth  
✅ Prefer **quality over quantity** of trades  
✅ Comfortable with **fewer positions** (max 5)  
✅ Risk-averse personality  
✅ Building confidence in system  
✅ Small account size (<$5k)  

### Choose AGGRESSIVE if:

✅ Trading on **testnet** (paper trading)  
✅ Experimenting with strategies  
✅ Want **faster data collection**  
✅ Can tolerate 4-5% drawdown  
✅ Want **more trades** for analysis  
✅ Comfortable with **more positions** (max 10)  
✅ Growth-oriented mindset  
✅ Larger account size (>$10k)  
✅ Strong market conditions  

---

## 🔧 QUICK COMMANDS

### Switch to SAFE (Default)

```powershell
Remove-Item env:ORCH_PROFILE -ErrorAction SilentlyContinue
docker-compose restart backend
python check_profile_status.py
```

### Switch to AGGRESSIVE

```powershell
$env:ORCH_PROFILE="AGGRESSIVE"
docker-compose restart backend
python check_profile_status.py
```

### Check Current Profile

```powershell
python check_profile_status.py
```

### Compare Profiles

```powershell
python compare_profiles.py
```

### Verify Backend

```powershell
docker logs quantum_backend | Select-String "profile"
# Expected: "🛡️ Loading SAFE profile" or "⚡ Loading AGGRESSIVE profile"
```

---

## 📊 EXPECTED RESULTS (24-Hour Sample)

### SAFE Profile Results

```
Trades Executed:      5-10 per day
Average Position:     $50-$80 (on $10k account)
Max Drawdown:         1.5-2.5%
Win Rate Required:    45%+
Symbols Traded:       Top quality only (3-5 symbols)
Exit Speed:           Fast in adverse conditions
Recovery:             Slow and steady
```

### AGGRESSIVE Profile Results

```
Trades Executed:      10-20 per day
Average Position:     $100-$150 (on $10k account)
Max Drawdown:         3.0-4.5%
Win Rate Required:    35%+
Symbols Traded:       More variety (5-10 symbols)
Exit Speed:           Trend-following everywhere
Recovery:             Fast after single win
```

---

## ⚠️ IMPORTANT WARNINGS

### SAFE Profile

⚠️ **May miss opportunities** in strong bull markets  
⚠️ **Slow recovery** after losing streaks (2 wins needed)  
⚠️ **Limited exposure** (max 10% of capital)  
⚠️ **Conservative exits** may cut winners short  

### AGGRESSIVE Profile

⚠️ **Higher drawdown risk** (up to 4.5%)  
⚠️ **More losing trades** (lower confidence threshold)  
⚠️ **Larger position sizes** can amplify mistakes  
⚠️ **Accepts lower quality symbols** (35% WR)  
⚠️ **NOT RECOMMENDED** for real capital initially  

---

## 🎓 BEST PRACTICES

### Starting Out

1. **Always start with SAFE** on real capital
2. **Use AGGRESSIVE on testnet** to test strategies
3. **Monitor for 1-2 weeks** before switching
4. **Compare results** with check_profile_status.py
5. **Document changes** in trading journal

### Switching Profiles

1. **Never switch mid-session** (close all positions first)
2. **Always restart backend** after env var change
3. **Verify profile loaded** with logs
4. **Test on testnet first**
5. **Monitor closely** for first 24 hours

### Monitoring

1. **Check daily drawdown** vs profile limit
2. **Monitor position sizes** for correctness
3. **Verify confidence thresholds** in logs
4. **Track trade frequency** (SAFE < AGG)
5. **Review exit modes** match expectations

---

## 📞 TROUBLESHOOTING

### Profile Not Loading?

```powershell
# Check environment variable
echo $env:ORCH_PROFILE

# Reset to SAFE
Remove-Item env:ORCH_PROFILE -ErrorAction SilentlyContinue

# Restart
docker-compose restart backend
```

### Wrong Parameters?

```powershell
# Check active profile
python check_profile_status.py

# Should show all parameters clearly
```

### Need Custom Profile?

```python
# Edit: backend/services/orchestrator_config.py
# Add new profile:
CUSTOM_PROFILE = {
    "base_confidence": 0.50,  # Your value
    "base_risk_pct": 1.0,     # Your value
    # ... copy rest from SAFE or AGGRESSIVE
}

# Update load_profile() function to include "CUSTOM"
```

---

## ✅ SUMMARY

```
┌────────────────────────────────────────────────────────────┐
│                   PROFILE COMPARISON                       │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  🛡️  SAFE = Conservative, Selective, Protective           │
│     • 0.8% base risk                                       │
│     • 55% confidence (fewer trades)                        │
│     • 2.5% DD limit                                        │
│     • 5 max positions                                      │
│     • Defensive exits                                      │
│     ✓ For REAL CAPITAL                                     │
│                                                            │
│  ⚡ AGGRESSIVE = Growth, Active, Experimental             │
│     • 1.2% base risk                                       │
│     • 45% confidence (more trades)                         │
│     • 4.5% DD limit                                        │
│     • 10 max positions                                     │
│     • Trend-following always                               │
│     ✓ For TESTNET                                          │
│                                                            │
└────────────────────────────────────────────────────────────┘

Switch with: $env:ORCH_PROFILE="SAFE" or "AGGRESSIVE"
Then restart: docker-compose restart backend
```

---

**🎯 Choose wisely, trade safely! 🎯**
