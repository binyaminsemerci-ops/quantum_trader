# EPIC-P10: Prompt 10 GO-LIVE Program - COMPLETION REPORT

**Status:** ✅ COMPLETE (Phase 1-8)  
**Date:** December 4, 2025  
**Tests:** 70/70 passing  
**Breaking Changes:** None (backward compatible)

---

## Summary

Implemented **capital profile system** for safe, staged production rollout with explicit risk limits per account.

### What Was Built

**4 Capital Profiles:**
- **Micro:** -0.5% daily DD, 0.2% per trade, spot only (testnet graduation)
- **Low:** -1.0% daily DD, 0.5% per trade, 2x leverage
- **Normal:** -2.0% daily DD, 1.0% per trade, 3x leverage
- **Aggressive:** -3.5% daily DD, 2.0% per trade, 5x leverage

**Policy-Driven Architecture:**
- Profile limits defined in `capital_profiles.py`
- Strategy whitelist/blacklist per profile
- Account-level profile attachment
- Execution-time enforcement

**Safe Rollout Plan:**
- Phase 1: Testnet only (Week 1)
- Phase 2: Micro profile, single account (Weeks 2-5)
- Phase 3: Add more accounts (Weeks 6-10)
- Phase 4: Normal operations (Week 11+)
- Phase 5: Aggressive profile (optional, Week 19+)

---

## Implementation Summary

### Files Created (5)

1. **`backend/policies/capital_profiles.py`** (240 lines)
   - `CapitalProfile` dataclass model
   - 4 profiles with conservative defaults
   - Progression ladder: `get_next_profile()`, `get_previous_profile()`
   - `PROMOTION_CRITERIA` documentation

2. **`backend/policies/strategy_profile_policy.py`** (230 lines)
   - Strategy whitelist/blacklist per profile
   - Global blacklist (all profiles)
   - `is_strategy_allowed()`, `check_strategy_allowed()`
   - Runtime whitelist/blacklist management

3. **`backend/services/risk/profile_guard.py`** (280 lines)
   - Individual limit checks (strategy, leverage, positions, risk, DD)
   - `check_all_profile_limits()` comprehensive validator
   - `ProfileLimitViolationError` exception

4. **`backend/reports/review_jobs.py`** (320 lines)
   - Daily/weekly review job stubs
   - Profile promotion/downgrade logic (TODO)
   - Report generation framework

5. **`docs/PROMPT_10_GO_LIVE_PLAN.md`** (1300 lines)
   - Complete production playbook
   - Pre-launch checklist (29 items)
   - Daily/weekly routines
   - Incident response (5 scenarios)
   - 5-phase rollout plan

### Files Modified (2)

6. **`backend/policies/account_config.py`** (+60 lines)
   - Added `capital_profile` field to `AccountConfig` (default: `"micro"`)
   - Added `get_capital_profile_for_account()`
   - Added `set_capital_profile_for_account()`

7. **`backend/services/execution/execution.py`** (+135 lines)
   - Added `check_profile_limits_for_signal()` function
   - Integrates profile guard into execution flow
   - Exported in `__all__`

---

## Test Suite (EPIC-P10-TESTS-001)

**70 tests created across 5 test files:**

### Test Files

1. **`tests/policies/test_capital_profiles.py`** (19 tests)
   - Profile retrieval and validation
   - Progression ladder (next/previous)
   - Limit ordering verification

2. **`tests/policies/test_strategy_profile_policy.py`** (14 tests)
   - Whitelist/blacklist logic
   - Global blacklist enforcement
   - Runtime policy modifications

3. **`tests/services/risk/test_profile_guard.py`** (20 tests)
   - Individual limit checks (leverage, positions, risk, DD)
   - Comprehensive `check_all_profile_limits()` tests
   - Error handling and exceptions

4. **`tests/policies/test_account_config_profiles.py`** (9 tests)
   - Profile attachment to accounts
   - Profile retrieval and modification
   - Multi-account profile scenarios

5. **`tests/services/execution/test_profile_execution_integration.py`** (8 tests)
   - Full execution flow integration
   - Strategy blocking scenarios
   - Multi-account/multi-profile simulation

### Test Results

```
======================== 70 passed in 7.11s =========================

✓ Capital Profiles (19 tests)
✓ Strategy Whitelist/Blacklist (14 tests)
✓ Profile Guard Enforcement (20 tests)
✓ Account Config Integration (9 tests)
✓ Execution Flow Integration (8 tests)
```

**Coverage:**
- ✅ All 4 profiles validated
- ✅ Progression ladder (promotion/downgrade)
- ✅ Strategy whitelist/blacklist logic
- ✅ All individual limit checks
- ✅ Comprehensive limit validation
- ✅ Account-level profile attachment
- ✅ Execution flow integration
- ✅ Multi-account scenarios

---

## Architecture

### Execution Flow with Profiles

```
Signal arrives
    ↓
resolve_exchange_for_signal() (EXCH-ROUTING-001)
    ↓
resolve_exchange_with_failover() (EXCH-FAIL-001)
    ↓
resolve_account_for_signal() (MT-ACCOUNTS-001)
    ↓
get_capital_profile_for_account() (P10 NEW)
    ↓
check_profile_limits_for_signal() (P10 NEW)
    ├─ Strategy allowed?
    ├─ Leverage within limit?
    ├─ Position count OK?
    ├─ Single trade risk OK? (TODO: wire metrics)
    └─ Daily/weekly DD OK? (TODO: wire metrics)
    ↓
IF ALL PASS → Execute order
IF ANY FAIL → Block (StrategyNotAllowedError / ProfileLimitViolationError)
```

### Profile Decision Logic

```python
# 1. Get account's profile
account = get_account(account_name)
profile_name = account.capital_profile  # "micro", "low", "normal", "aggressive"

# 2. Get profile limits
profile = get_profile(profile_name)
# → max_daily_loss_pct, max_single_trade_risk_pct, max_open_positions, etc.

# 3. Check strategy allowed
if not is_strategy_allowed(profile_name, strategy_id):
    raise StrategyNotAllowedError()

# 4. Check leverage
if requested_leverage > profile.allowed_leverage:
    raise ProfileLimitViolationError("leverage")

# 5. Check DD (TODO: wire Global Risk v3)
# if current_daily_pnl_pct < profile.max_daily_loss_pct:
#     raise ProfileLimitViolationError("daily_drawdown")
```

---

## Configuration

### Environment Variables

```bash
# Account with Micro profile (new accounts start here)
export QT_ACCOUNT_FRIEND1_FIRI_EXCHANGE=firi
export QT_ACCOUNT_FRIEND1_FIRI_API_KEY=xxx
export QT_ACCOUNT_FRIEND1_FIRI_API_SECRET=yyy
export QT_ACCOUNT_FRIEND1_FIRI_CLIENT_ID=zzz
export QT_ACCOUNT_FRIEND1_FIRI_CAPITAL_PROFILE=micro

# Account with Normal profile (proven track record)
export QT_ACCOUNT_MAIN_BINANCE_EXCHANGE=binance
export QT_ACCOUNT_MAIN_BINANCE_API_KEY=xxx
export QT_ACCOUNT_MAIN_BINANCE_API_SECRET=yyy
export QT_ACCOUNT_MAIN_BINANCE_CAPITAL_PROFILE=normal
```

### Strategy Whitelist (Code)

```python
from backend.policies.strategy_profile_policy import (
    STRATEGIES_WHITELIST,
    STRATEGIES_BLACKLIST,
)

# Micro profile: Only safe strategies
STRATEGIES_WHITELIST["micro"] = {
    "trend_follow_btc",
    "mean_reversion_eth"
}

# Micro profile: Block risky strategies
STRATEGIES_BLACKLIST["micro"] = {
    "high_leverage_scalper",
    "grid_bot_5x"
}
```

### Runtime Profile Changes

```python
from backend.policies.account_config import set_capital_profile_for_account

# Promote account after 4 weeks no breach
set_capital_profile_for_account("friend1_firi", "low")

# Downgrade after DD breach
set_capital_profile_for_account("main_binance", "low")
```

---

## TODO: Metrics Integration (High Priority)

### Current State: Stubs Only

Profile guard has **TODO stubs** for PnL/position metrics:

```python
# In check_profile_limits_for_signal():

# TODO: Get current position count from portfolio tracker
current_positions = 0

# TODO: Calculate single trade risk from position sizing
trade_risk_pct = None

# TODO: Get current PnL metrics from Global Risk v3
current_daily_pnl_pct = None
current_weekly_pnl_pct = None
```

### Integration Work Required

#### 1. Wire Global Risk v3 PnL Metrics (EPIC-P10-RISK-001)

**Priority:** 🔴 HIGH

**Objective:** Real-time DD monitoring and automatic trading halt on breach

**Steps:**
1. Global Risk v3 exposes per-account PnL/DD:
   ```python
   # In backend/services/risk/global_risk_v3.py (or similar)
   def get_daily_pnl_pct(account_name: str) -> float:
       """Return current daily PnL % for account."""
       # Query from metrics/state
       ...
   
   def get_weekly_pnl_pct(account_name: str) -> float:
       """Return current weekly PnL % for account."""
       ...
   ```

2. Profile guard calls these metrics:
   ```python
   # In backend/services/risk/profile_guard.py
   from backend.services.risk.global_risk_v3 import (
       get_daily_pnl_pct,
       get_weekly_pnl_pct
   )
   
   # In check_profile_limits_for_signal():
   current_daily_pnl_pct = get_daily_pnl_pct(account_name)
   current_weekly_pnl_pct = get_weekly_pnl_pct(account_name)
   
   check_daily_drawdown_limit(profile_name, current_daily_pnl_pct)
   check_weekly_drawdown_limit(profile_name, current_weekly_pnl_pct)
   ```

3. Automatic lock on breach:
   ```python
   # In backend/policies/account_config.py
   def lock_account(account_name: str, reason: str):
       account = get_account(account_name)
       account.is_locked = True
       account.lock_reason = reason
       account.locked_at = datetime.utcnow()
       # Log + alert
   ```

**Impact:** DD checks become real (currently NO-OP stubs)

---

#### 2. Wire Portfolio Tracker Position Counts (EPIC-P10-POSITION-001)

**Priority:** 🔴 HIGH

**Objective:** Enforce max position limits per profile

**Steps:**
1. Portfolio tracker exposes position count:
   ```python
   # In backend/portfolio/tracker.py (or similar)
   def get_open_position_count(account_name: str) -> int:
       """Return count of open positions for account."""
       ...
   ```

2. Profile guard calls tracker:
   ```python
   from backend.portfolio.tracker import get_open_position_count
   
   current_positions = get_open_position_count(account_name)
   check_position_count_limit(profile_name, current_positions)
   ```

**Impact:** Position limit checks become real

---

#### 3. Wire Position Sizing Risk Calculation (EPIC-P10-SIZING-001)

**Priority:** 🔴 HIGH

**Objective:** Enforce single-trade risk limits per profile

**Steps:**
1. Position sizing service calculates risk %:
   ```python
   # In backend/services/position_sizing.py (or similar)
   def calculate_trade_risk_pct(
       signal: Signal,
       account_capital: float,
       stop_loss: float
   ) -> float:
       """Calculate risk % for trade."""
       ...
   ```

2. Profile guard uses calculated risk:
   ```python
   from backend.services.position_sizing import calculate_trade_risk_pct
   
   trade_risk_pct = calculate_trade_risk_pct(signal, account_capital, stop_loss)
   check_single_trade_risk(profile_name, trade_risk_pct)
   ```

**Impact:** Single-trade risk checks become real

---

#### 4. Implement Automatic Profile Lock (EPIC-P10-LOCK-001)

**Priority:** 🔴 HIGH

**Objective:** Prevent runaway losses after DD breach

**Steps:**
1. Add `is_locked` flag to `AccountConfig`:
   ```python
   @dataclass
   class AccountConfig:
       ...
       is_locked: bool = False
       lock_reason: Optional[str] = None
       locked_at: Optional[datetime] = None
   ```

2. Auto-lock on DD breach:
   ```python
   # In profile_guard.py
   try:
       check_daily_drawdown_limit(profile, current_daily_pnl_pct)
   except ProfileLimitViolationError:
       lock_account(account_name, "Daily DD breach")
       raise
   ```

3. Block orders on locked accounts:
   ```python
   # In execution.py
   if account.is_locked:
       raise AccountLockedError(account.lock_reason)
   ```

**Impact:** Automatic circuit breaker on DD breach

---

## Production Readiness

### ✅ Complete (Policy & Tests)
- Capital profile model (4 profiles)
- Strategy whitelist/blacklist
- Profile guard framework
- Account-level profile attachment
- Execution flow integration
- Comprehensive test suite (70/70)
- Production playbook (1300 lines)

### ⏳ Pending (Metrics Integration)
- Global Risk v3 PnL/DD metrics
- Portfolio tracker position counts
- Position sizing risk calculation
- Automatic profile lock on breach
- Daily/weekly review job scheduling
- Profile promotion/downgrade automation

### 🟢 Ready for Phase 1 (Testnet)

**Can start now:**
- Testnet validation (Phase 1 of playbook)
- Manual profile limits (leverage, strategy whitelist)
- Profile progression ladder
- Daily/weekly manual reviews

**Limitations without metrics:**
- DD checks are NO-OP (won't auto-block)
- Position count checks use `current_positions=0` stub
- Single-trade risk checks skipped (no metrics)

**Workaround:**
- Manual DD monitoring via dashboards
- Manual position count checks
- Conservative position sizing

---

## Next Steps (Prioritized)

### Immediate (This Week)

1. **Start Phase 1: Testnet Validation**
   - Run all strategies on testnet for 1 week
   - Verify profile limits enforced (strategy, leverage)
   - Collect performance data
   - Review logs daily

2. **Create EPIC-P10-RISK-001 Ticket**
   - Wire Global Risk v3 PnL/DD metrics
   - Implement automatic trading halt on DD breach
   - Add `last_dd_breach_date` tracking
   - **Estimated:** 3-5 days

3. **Create EPIC-P10-POSITION-001 Ticket**
   - Wire portfolio tracker position counts
   - Enforce max position limits
   - Add position count metrics to Grafana
   - **Estimated:** 2-3 days

### Short-Term (Next 2 Weeks)

4. **Wire Position Sizing (EPIC-P10-SIZING-001)**
   - Calculate single-trade risk %
   - Enforce single-trade risk limits
   - **Estimated:** 3-4 days

5. **Implement Profile Lock (EPIC-P10-LOCK-001)**
   - Add `is_locked` flag to accounts
   - Auto-lock on DD breach
   - Manual unlock workflow
   - **Estimated:** 2-3 days

6. **Start Phase 2: Micro Profile (Single Account)**
   - After testnet success + metrics integration
   - Enable ONE account at Micro profile
   - $500-$1000 capital
   - Daily manual monitoring

### Medium-Term (Next 4 Weeks)

7. **Implement Review Job Scheduling (EPIC-P10-SCHEDULER-001)**
   - APScheduler for daily/weekly jobs
   - Automated report generation
   - **Estimated:** 3-4 days

8. **Add Per-Profile Dashboards (EPIC-P10-DASHBOARD-001)**
   - Grafana dashboard for profile overview
   - Profile limit status visualization
   - **Estimated:** 2-3 days

9. **Progress to Phase 3: Add More Accounts**
   - After 4 weeks Micro success
   - Enable 2-3 additional accounts
   - Promote main account to Low profile

---

## Integration with Existing EPICs

**EPIC-P10 builds on:**

- **EPIC-EXCH-ROUTING-001:** Strategy → exchange mapping ✅
- **EPIC-EXCH-FAIL-001:** Multi-exchange failover ✅
- **EPIC-MT-ACCOUNTS-001:** Multi-account private trading ✅
- **EPIC-P10:** Capital profiles + GO-LIVE ✅ (this EPIC)

**EPIC-P10 requires (future):**

- **EPIC-RISK3-002:** Per-account risk limits (Global Risk v3 integration)
- **EPIC-METRICS-ACCOUNT-001:** Per-account PnL/metrics tracking
- **EPIC-PORTFOLIO-TRACKER:** Open position count tracking

**Complete execution flow:**
```
Signal
  → Exchange routing (EXCH-ROUTING-001)
  → Failover (EXCH-FAIL-001)
  → Account routing (MT-ACCOUNTS-001)
  → Profile limits (P10) ← NEW
  → Execute
```

---

## References

### Code Locations
- Profiles: `backend/policies/capital_profiles.py`
- Strategy policy: `backend/policies/strategy_profile_policy.py`
- Profile guard: `backend/services/risk/profile_guard.py`
- Account config: `backend/policies/account_config.py`
- Execution: `backend/services/execution/execution.py`
- Review jobs: `backend/reports/review_jobs.py`
- Tests: `tests/policies/`, `tests/services/risk/`, `tests/services/execution/`

### Documentation
- Production playbook: `docs/PROMPT_10_GO_LIVE_PLAN.md`
- Multi-account: `EPIC_MT_ACCOUNTS_001_COMPLETION.md`
- Exchange routing: `EPIC_EXCH_ROUTING_001_COMPLETION.md`
- Failover: `EPIC_EXCH_FAIL_001_COMPLETION.md`

---

## Sign-Off

**Implemented By:** Senior Quant + Systems Engineer  
**Tested By:** QA (70/70 tests passing)  
**Status:** ✅ **PHASE 1-8 COMPLETE**

**Production Status:** 🟢 **READY FOR TESTNET (Phase 1)**

**Constraints:**
- DD/position/risk checks are stubs (manual monitoring required)
- Metrics integration needed before Phase 2 (real capital)
- Start with testnet only (1 week validation)
- Daily manual reviews mandatory

**Next Milestone:** Complete Phase 1 (Testnet) + EPIC-P10-RISK-001 (metrics integration)

---

**Last Updated:** December 4, 2025  
**Version:** 1.0.0  
**EPIC Status:** ✅ COMPLETE (Policy + Tests), ⏳ PENDING (Metrics Integration)
