# EPIC-P10 Quick Reference

**Status:** ✅ 70/70 tests passing | 🟢 Ready for Phase 1 (Testnet)

---

## What You Got

✅ **4 Capital Profiles** (Micro/Low/Normal/Aggressive)  
✅ **Strategy Whitelist/Blacklist** per profile  
✅ **Profile Guard** (leverage, positions, strategy checks)  
✅ **Account-Level Profiles** (default: Micro)  
✅ **70 Tests** (comprehensive coverage)  
✅ **Production Playbook** (1300 lines, 5-phase rollout)

---

## Quick Start

### 1. Configure Account with Profile

```bash
# Friend account (Micro profile for safety)
export QT_ACCOUNT_FRIEND1_FIRI_EXCHANGE=firi
export QT_ACCOUNT_FRIEND1_FIRI_API_KEY=xxx
export QT_ACCOUNT_FRIEND1_FIRI_API_SECRET=yyy
export QT_ACCOUNT_FRIEND1_FIRI_CLIENT_ID=zzz
export QT_ACCOUNT_FRIEND1_FIRI_CAPITAL_PROFILE=micro

# Main account (Normal profile - proven)
export QT_ACCOUNT_MAIN_BINANCE_CAPITAL_PROFILE=normal
```

### 2. Set Strategy Whitelist (Optional)

```python
from backend.policies.strategy_profile_policy import (
    STRATEGIES_WHITELIST,
    STRATEGIES_BLACKLIST
)

# Micro: Only safe strategies
STRATEGIES_WHITELIST["micro"] = {
    "trend_follow_btc",
    "mean_reversion_eth"
}

# Block risky strategies
STRATEGIES_BLACKLIST["micro"] = {
    "high_leverage_scalper"
}
```

### 3. Execution Flow

Profile checks happen automatically:

```python
# In execution service (automatic)
account_name = resolve_account_for_signal(...)
check_profile_limits_for_signal(
    account_name,
    strategy_id,
    requested_leverage
)
# → Blocks if strategy blacklisted or leverage too high
```

---

## Profile Limits

| Profile | Daily DD | Weekly DD | Trade Risk | Max Positions | Leverage |
|---------|----------|-----------|------------|---------------|----------|
| **Micro** | -0.5% | -2.0% | 0.2% | 2 | 1x |
| **Low** | -1.0% | -3.5% | 0.5% | 3 | 2x |
| **Normal** | -2.0% | -7.0% | 1.0% | 5 | 3x |
| **Aggressive** | -3.5% | -12.0% | 2.0% | 8 | 5x |

---

## Test Results

```bash
python -m pytest tests/policies/ tests/services/risk/test_profile_guard.py \
  tests/services/execution/test_profile_execution_integration.py -v
```

**Result:** 70 passed in 7.11s ✅

---

## What Works Now

✅ Strategy blacklist enforcement  
✅ Leverage limit checks  
✅ Account-level profile attachment  
✅ Runtime profile changes  
✅ Progression ladder (micro→low→normal→aggressive)

---

## What's Stubbed (TODO)

⏳ Daily/weekly DD checks (need Global Risk v3 PnL metrics)  
⏳ Position count checks (need portfolio tracker)  
⏳ Single-trade risk checks (need position sizing)  
⏳ Automatic profile lock on DD breach  
⏳ Daily/weekly review job automation

---

## Phase 1: Start Testnet (Now)

### Pre-Flight Checklist

- [ ] All accounts configured with profiles
- [ ] Strategy whitelist/blacklist set
- [ ] Backend service running
- [ ] Testnet mode enabled
- [ ] Observability dashboards ready

### Daily Routine (Phase 1)

**Morning (9:00 AM):**
- Check backend health
- Review overnight trades (if any)
- Verify profile limits respected

**Evening (6:00 PM):**
- Review daily PnL
- Check for errors in logs
- Verify strategy routing correct

### Success Criteria (1 Week)

✅ No critical errors  
✅ All orders executed correctly  
✅ Profile limits respected  
✅ Strategy whitelist/blacklist working

---

## Next: Metrics Integration

### EPIC-P10-RISK-001 (High Priority)

**Wire Global Risk v3 PnL metrics:**

```python
# Add to Global Risk v3:
def get_daily_pnl_pct(account_name: str) -> float:
    ...

def get_weekly_pnl_pct(account_name: str) -> float:
    ...

# Profile guard calls:
current_daily_pnl = get_daily_pnl_pct(account_name)
check_daily_drawdown_limit(profile, current_daily_pnl)
```

**Impact:** DD checks become real (currently NO-OP)

### EPIC-P10-POSITION-001 (High Priority)

**Wire portfolio tracker:**

```python
# Portfolio tracker exposes:
def get_open_position_count(account_name: str) -> int:
    ...

# Profile guard calls:
current_positions = get_open_position_count(account_name)
check_position_count_limit(profile, current_positions)
```

**Impact:** Position limit checks become real

---

## Manual Overrides

### Promote Account

```python
from backend.policies.account_config import set_capital_profile_for_account

# After 4 weeks Micro success → Low
set_capital_profile_for_account("friend1_firi", "low")
```

### Downgrade Account

```python
# After DD breach → Downgrade
set_capital_profile_for_account("main_binance", "low")
```

### Block Strategy

```python
from backend.policies.strategy_profile_policy import add_strategy_to_blacklist

# Block strategy for profile
add_strategy_to_blacklist("micro", "risky_strategy")
```

---

## Key Files

**Policies:**
- `backend/policies/capital_profiles.py` - Profile definitions
- `backend/policies/strategy_profile_policy.py` - Whitelist/blacklist
- `backend/policies/account_config.py` - Account profiles

**Enforcement:**
- `backend/services/risk/profile_guard.py` - Limit checks
- `backend/services/execution/execution.py` - Integration

**Documentation:**
- `docs/PROMPT_10_GO_LIVE_PLAN.md` - Full playbook
- `EPIC_P10_COMPLETION.md` - Detailed completion report

**Tests:**
- `tests/policies/test_capital_profiles.py` (19 tests)
- `tests/policies/test_strategy_profile_policy.py` (14 tests)
- `tests/services/risk/test_profile_guard.py` (20 tests)
- `tests/policies/test_account_config_profiles.py` (9 tests)
- `tests/services/execution/test_profile_execution_integration.py` (8 tests)

---

## Production Checklist (Before Real Capital)

**Infrastructure:**
- [ ] Exchanges healthy
- [ ] Multi-exchange failover tested
- [ ] Account routing verified
- [ ] Backend service stable

**Risk:**
- [ ] Global Risk v3 operational
- [ ] Capital profiles configured
- [ ] Strategy policies defined
- [ ] DD monitoring ready

**Observability:**
- [ ] Grafana dashboards
- [ ] Prometheus metrics
- [ ] Log aggregation
- [ ] Alerts configured

**Metrics Integration (Critical):**
- [ ] EPIC-P10-RISK-001 complete (DD metrics)
- [ ] EPIC-P10-POSITION-001 complete (position tracking)
- [ ] EPIC-P10-SIZING-001 complete (risk calculation)

---

## Emergency Procedures

### Stop Trading

```bash
# Kill backend
pkill -f "uvicorn backend.main:app"

# OR use kill switch
curl -X POST http://localhost:8000/admin/kill-switch
```

### Lock Account

```python
from backend.policies.account_config import get_account

account = get_account("main_binance")
account.is_locked = True  # TODO: Add this field
```

### Downgrade Profile

```python
set_capital_profile_for_account("main_binance", "micro")
```

---

## Resources

**Full Documentation:**
- Production playbook: `docs/PROMPT_10_GO_LIVE_PLAN.md`
- Completion report: `EPIC_P10_COMPLETION.md`

**Related EPICs:**
- Exchange routing: `EPIC_EXCH_ROUTING_001_COMPLETION.md`
- Failover: `EPIC_EXCH_FAIL_001_COMPLETION.md`
- Multi-account: `EPIC_MT_ACCOUNTS_001_COMPLETION.md`

**Dashboards:**
- Grafana: http://localhost:3000
- Backend health: http://localhost:8000/health

---

**Ready to Start Phase 1! 🚀**

**Default:** All new accounts start at **Micro** profile (ultra-safe)  
**Promotion:** Manual after 4 weeks no breach + good metrics  
**Monitoring:** Daily reviews mandatory during Phase 1-3

---

**Last Updated:** December 4, 2025
