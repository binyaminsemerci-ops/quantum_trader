# EPIC-MT-ACCOUNTS-001: Multi-Account Private Trading

**Status:** ✅ COMPLETE  
**Date:** December 4, 2025  
**Tests:** 17/17 passing  
**Breaking Changes:** None (backward compatible)

---

## Summary

### How Multiple Private Accounts Work

Quantum Trader v2.0 now supports **private multi-account trading** - allowing you to trade for multiple accounts (yours + close friends/family) in the same instance:

```python
# Account 1: Your main Binance account
AccountConfig(
    name="main_binance",
    exchange="binance",
    api_key=os.getenv("BINANCE_API_KEY"),
    api_secret=os.getenv("BINANCE_API_SECRET")
)

# Account 2: Friend's Firi account
AccountConfig(
    name="friend_1_firi",
    exchange="firi",
    api_key=os.getenv("FRIEND1_FIRI_API_KEY"),
    api_secret=os.getenv("FRIEND1_FIRI_SECRET_KEY"),
    client_id=os.getenv("FRIEND1_FIRI_CLIENT_ID")
)
```

### Account Selection Per Order

**3-tier routing priority:**

```
1. signal.account_name (explicit) → Use immediately
       ↓
2. STRATEGY_ACCOUNT_MAP[strategy_id] (policy) → Lookup
       ↓
3. "main_<exchange>" (default) → Safe fallback
```

**Example flow:**

```python
# Step 1: Resolve exchange (EXCH-ROUTING-001)
exchange = resolve_exchange_for_signal(
    signal.exchange,
    signal.strategy_id
)
# Returns: "firi"

# Step 2: Resolve account (MT-ACCOUNTS-001)
account_name = resolve_account_for_signal(
    signal.account_name,
    signal.strategy_id,
    exchange
)
# Returns: "friend_1_firi"

# Step 3: Get account client
account = get_account(account_name)
client = get_exchange_client_for_account(account)
# Returns: FiriAdapter with friend's credentials

# Step 4: Execute order
result = await client.place_order(order_request)
```

### Backward Compatibility

✅ **No breaking changes:**
- Existing single-account setup works unchanged
- Legacy env vars (`BINANCE_API_KEY`, `FIRI_API_KEY`) auto-register as `main_<exchange>` accounts
- All new fields (`signal.account_name`) are `Optional`
- Default behavior: Routes to `main_<exchange>` if no account specified

---

## Implementation

### Files Created (4)

1. **`backend/policies/account_config.py`** (310 lines)
   - `AccountConfig` dataclass — Account credentials + metadata
   - `ACCOUNTS` registry — In-memory account storage
   - `register_account()`, `get_account()`, `list_accounts()`
   - `get_default_account_for_exchange()`
   - `load_accounts_from_env()` — Auto-register from `QT_ACCOUNT_*` env vars
   - `register_legacy_accounts()` — Backward compatibility

2. **`backend/policies/account_mapping.py`** (90 lines)
   - `STRATEGY_ACCOUNT_MAP` — Strategy → account mapping
   - `get_account_for_strategy()` — Account selection logic
   - `set_strategy_account_mapping()` — Runtime configuration

3. **`backend/services/risk/account_limits.py`** (55 lines)
   - `check_account_limits()` — Placeholder for per-account risk controls
   - NO-OP for now, ready for Global Risk v3 integration

4. **`tests/services/execution/test_multi_account_routing.py`** (280 lines)
   - 17 test cases covering all routing scenarios
   - Unit tests: account registration, mapping, factory integration
   - Integration tests: full signal → exchange → account flow

### Files Updated (3)

5. **`backend/integrations/exchanges/factory.py`** (+120 lines)
   - `get_exchange_config_for_account()` — Convert AccountConfig → ExchangeConfig
   - `get_exchange_client_for_account()` — Main entry point for multi-account
   - Exported in `__all__`

6. **`backend/routes/signals.py`** (+5 lines)
   - Extended `Signal` model: `account_name: Optional[str]`

7. **`backend/services/execution/execution.py`** (+100 lines)
   - `resolve_account_for_signal()` — Account routing decision logic
   - Exported in `__all__`

---

## Configuration

### Environment Variables (New Format)

```bash
# Account 1: Main Binance
export QT_ACCOUNT_MAIN_BINANCE_EXCHANGE=binance
export QT_ACCOUNT_MAIN_BINANCE_API_KEY=xxx
export QT_ACCOUNT_MAIN_BINANCE_API_SECRET=yyy
export QT_ACCOUNT_MAIN_BINANCE_TESTNET=false
export QT_ACCOUNT_MAIN_BINANCE_MODE=real
export QT_ACCOUNT_MAIN_BINANCE_DESCRIPTION="Primary Binance account"

# Account 2: Friend's Firi
export QT_ACCOUNT_FRIEND1_FIRI_EXCHANGE=firi
export QT_ACCOUNT_FRIEND1_FIRI_API_KEY=xxx
export QT_ACCOUNT_FRIEND1_FIRI_API_SECRET=yyy
export QT_ACCOUNT_FRIEND1_FIRI_CLIENT_ID=zzz
export QT_ACCOUNT_FRIEND1_FIRI_MODE=real
export QT_ACCOUNT_FRIEND1_FIRI_DESCRIPTION="Friend 1 - NOK trading"
```

### Legacy Format (Still Supported)

```bash
# Auto-registers as "main_binance"
export BINANCE_API_KEY=xxx
export BINANCE_API_SECRET=yyy

# Auto-registers as "main_firi"
export FIRI_API_KEY=xxx
export FIRI_SECRET_KEY=yyy
export FIRI_CLIENT_ID=zzz
```

### Strategy → Account Mapping

```python
from backend.policies.account_mapping import set_strategy_account_mapping

set_strategy_account_mapping({
    "scalper_btc": "main_binance",
    "swing_eth": "main_binance",
    "nordic_spot": "main_firi",
    "friend_1_strategy": "friend_1_firi",
})
```

---

## Usage Examples

### Example 1: Explicit Account Override

```python
# AI Engine sends signal with explicit account
signal = Signal(
    symbol="BTC/USDT",
    side="long",
    score=0.85,
    confidence=0.92,
    account_name="friend_1_binance",  # ← Explicit override
    details={...}
)

# Execution Service:
account_name = resolve_account_for_signal(
    signal.account_name,  # "friend_1_binance"
    signal.strategy_id,
    exchange
)
# → Returns "friend_1_binance"

account = get_account("friend_1_binance")
client = get_exchange_client_for_account(account)
# → FiriAdapter with friend's API keys
```

### Example 2: Strategy-Based Account Routing

```python
# Set policy mapping
set_strategy_account_mapping({
    "friend_1_strategy": "friend_1_firi"
})

# Signal without explicit account
signal = Signal(
    ...,
    strategy_id="friend_1_strategy",  # ← Policy lookup
    account_name=None
)

# Execution Service:
account_name = resolve_account_for_signal(
    None,
    "friend_1_strategy",
    "firi"
)
# → Returns "friend_1_firi" (from policy)
```

### Example 3: Default Account Fallback

```python
# Signal without account or strategy
signal = Signal(
    ...,
    account_name=None,
    strategy_id=None
)

# Execution Service resolves exchange
exchange = resolve_exchange_for_signal(None, None)
# → "binance" (DEFAULT_EXCHANGE)

account_name = resolve_account_for_signal(None, None, "binance")
# → "main_binance" (default account for exchange)
```

### Example 4: Full Integrated Flow

```python
# Complete order execution flow with multi-account
from backend.services.execution.execution import (
    resolve_exchange_for_signal,
    resolve_account_for_signal
)
from backend.integrations.exchanges.factory import get_exchange_client_for_account
from backend.policies.account_config import get_account
from backend.services.risk.account_limits import check_account_limits

# Step 1: Resolve exchange (EXCH-ROUTING-001)
primary_exchange = resolve_exchange_for_signal(
    signal.exchange,
    signal.strategy_id
)

# Step 2: Apply failover if needed (EXCH-FAIL-001)
exchange_name = await resolve_exchange_with_failover(
    primary_exchange,
    "binance"
)

# Step 3: Resolve account (MT-ACCOUNTS-001)
account_name = resolve_account_for_signal(
    signal.account_name,
    signal.strategy_id,
    exchange_name
)

# Step 4: Get account and create client
account = get_account(account_name)
client = get_exchange_client_for_account(account)

# Step 5: Check account limits (placeholder)
check_account_limits(account_name, order_request)

# Step 6: Execute order
result = await client.place_order(order_request)
```

---

## Testing

### Test Results

```
====================== 17 passed in 7.15s =======================

✓ test_register_and_get_account
✓ test_get_account_not_found
✓ test_get_default_account_for_exchange
✓ test_get_default_account_no_accounts
✓ test_get_account_for_strategy_explicit_mapping
✓ test_get_account_for_strategy_default
✓ test_get_account_for_strategy_none
✓ test_get_exchange_config_for_account
✓ test_get_exchange_config_for_account_with_passphrase
✓ test_get_exchange_config_for_account_with_client_id
✓ test_get_exchange_config_invalid_exchange
✓ test_get_exchange_client_for_account_by_object
✓ test_get_exchange_client_for_account_by_name
✓ test_resolve_account_for_signal_explicit
✓ test_resolve_account_for_signal_strategy_mapping
✓ test_resolve_account_for_signal_default
✓ test_resolve_account_full_flow
```

### Test Coverage

- ✅ Account registration and retrieval
- ✅ Default account selection per exchange
- ✅ Strategy → account mapping
- ✅ AccountConfig → ExchangeConfig conversion
- ✅ Factory integration (by object and by name)
- ✅ Signal routing: explicit, policy, default
- ✅ Full flow: signal → exchange → account → client

### Run Tests

```powershell
python -m pytest tests/services/execution/test_multi_account_routing.py -v
```

---

## Monitoring

### Key Logs

```json
{
  "message": "Creating exchange client for account",
  "account_name": "friend_1_firi",
  "exchange": "firi",
  "testnet": false,
  "mode": "real",
  "level": "INFO"
}

{
  "message": "Using explicit account from signal",
  "account_name": "main_binance",
  "source": "signal.account_name",
  "strategy_id": "scalper_btc",
  "level": "INFO"
}

{
  "message": "Resolved account for signal",
  "account_name": "friend_1_firi",
  "strategy_id": "friend_1_strategy",
  "exchange": "firi",
  "source": "strategy_policy",
  "level": "INFO"
}
```

### Metrics to Track

```python
# Add to monitoring dashboard
account_orders_total{account_name="main_binance", exchange="binance"}
account_orders_total{account_name="friend_1_firi", exchange="firi"}

account_pnl_usd{account_name="main_binance"}
account_pnl_usd{account_name="friend_1_firi"}

account_exposure_usd{account_name="main_binance"}
account_exposure_usd{account_name="friend_1_firi"}
```

---

## Security & Privacy

### Credential Management

- **Environment variables:** Secure for private multi-account (not SaaS)
- **No database storage:** Credentials stay in env vars (private setup)
- **No encryption:** Suitable for private deployment (not multi-tenant)
- **Future:** Add KMS/vault integration if needed

### Account Isolation

- ✅ **Per-account API keys:** Each account uses own credentials
- ✅ **Independent execution:** Orders routed to correct account client
- ⏳ **Per-account risk limits:** Placeholder ready (check_account_limits)
- ⏳ **Per-account PnL tracking:** Future enhancement

### Audit Trail

- ✅ **Structured logging:** All account selections logged with context
- ✅ **Account name in logs:** Easy to filter by account
- ✅ **Source tracking:** Logs show if explicit/policy/default

---

## Limitations & Future Work

### Current Limitations

- **Config-driven only:** No database, no admin UI
- **Manual setup:** Env vars must be set manually
- **No per-account risk limits:** Placeholder only (NO-OP)
- **No per-account metrics:** Need separate PnL tracking
- **No account permissions:** All accounts have same capabilities

### Roadmap: TODO

#### 1. Per-Account Risk Limits (EPIC-RISK3-002)
- [ ] Implement `check_account_limits()` real logic
- [ ] Per-account exposure caps in PolicyStore/AccountConfig
- [ ] Per-account max leverage
- [ ] Per-account daily drawdown limits
- [ ] Per-account position limits

**Impact:** Prevent over-trading, protect each account independently

#### 2. Per-Account PnL & Metrics (EPIC-METRICS-ACCOUNT-001)
- [ ] Track PnL per account (separate from global PnL)
- [ ] Per-account exposure tracking
- [ ] Per-account trade history
- [ ] Per-account performance dashboard
- [ ] Per-account fee tracking

**Impact:** Clear visibility into each account's performance

#### 3. Basic Account Management CLI (EPIC-CLI-ACCOUNTS-001)
- [ ] CLI commands: `qt accounts list`, `qt accounts add`, `qt accounts remove`
- [ ] CLI command: `qt accounts test <name>` — Test API connection
- [ ] CLI command: `qt accounts balance <name>` — Check balance
- [ ] CLI command: `qt accounts status` — Show all account statuses

**Impact:** Easier account management without editing env vars

#### 4. Encryption/KMS for Secrets (EPIC-SEC-KMS-001)
- [ ] Encrypted storage for API keys
- [ ] AWS KMS / Azure Key Vault integration
- [ ] Rotation policy for API keys
- [ ] Audit log for key access

**Impact:** Enhanced security for production deployments

#### 5. Separate Paper vs Real Accounts (EPIC-ACCOUNTS-MODE-001)
- [ ] Enforce `mode="paper"` accounts only trade paper
- [ ] Validation: Prevent real orders on paper accounts
- [ ] Testing: Safe testing with friend accounts (paper mode)
- [ ] Dashboard: Visual indicator for paper vs real

**Impact:** Safe testing without risking real funds

#### 6. Account-Level Feature Flags (EPIC-ACCOUNTS-FEATURES-001)
- [ ] Per-account enabled exchanges
- [ ] Per-account allowed strategies
- [ ] Per-account max position size
- [ ] Per-account trading hours restrictions

**Impact:** Fine-grained control per account

---

## References

### Related EPICs

- **EPIC-EXCH-003:** Firi integration ✅
- **EPIC-EXCH-ROUTING-001:** Strategy → exchange mapping ✅
- **EPIC-EXCH-FAIL-001:** Multi-exchange failover ✅
- **EPIC-MT-ACCOUNTS-001:** Multi-account private trading ✅ (this document)
- **EPIC-RISK3-002:** Per-account risk limits ⏳
- **EPIC-METRICS-ACCOUNT-001:** Per-account PnL tracking ⏳

### Code Locations

- Account config: `backend/policies/account_config.py`
- Account mapping: `backend/policies/account_mapping.py`
- Factory integration: `backend/integrations/exchanges/factory.py`
- Execution integration: `backend/services/execution/execution.py`
- Risk stub: `backend/services/risk/account_limits.py`
- Tests: `tests/services/execution/test_multi_account_routing.py`

### Documentation

- [Exchange Routing](EPIC_EXCH_ROUTING_001_COMPLETION.md)
- [Exchange Failover](EPIC_EXCH_FAIL_001_COMPLETION.md)
- [Multi-Exchange Architecture](MULTI_EXCHANGE_QUICKREF.md)

---

## Sign-Off

**Implemented By:** Senior Backend Engineer  
**Reviewed By:** System Architecture  
**Tested By:** QA (17/17 tests passing)  
**Status:** ✅ **PRODUCTION READY**

**Deployment Note:** This is **private multi-account** for personal use + close friends/family. NOT multi-tenant SaaS. Credentials managed via environment variables suitable for private deployment.

**Next Steps:**
1. Configure accounts via `QT_ACCOUNT_*` env vars or legacy format
2. Set strategy → account mappings in `account_mapping.py`
3. Monitor account routing in logs (`account_name` field)
4. Plan EPIC-RISK3-002 (per-account risk limits)

---

**Last Updated:** December 4, 2025  
**Version:** 1.0.0
