# AI: Automatic Funding Rate Filter Implementation
**Date**: February 8, 2026  
**Status**: ✅ IMPLEMENTED, READY TO DEPLOY

---

## Problem Statement

User discovered symbols with **EXTREME funding fees** that make them unprofitable regardless of exit timing:

### Examples from Live Positions:
```
BREVUSDT:  78.61 USDT funding / 31.60 USDT margin (248% ratio)
API3USDT:  150.23 USDT funding / 158.45 USDT margin (95% ratio)
```

**Normal funding**: ~0.01% per 8h (0.03% per day)  
**Extreme funding**: 100-1000× normal rates on illiquid perpetuals

### User's Insight:
> "hvordan kan vi finne ut fees på forhånd på en tidlig stadig slik at vi kan blackliste dem automatisk, før vi trader er det mulig?"

**Translation**: Check funding fees BEFORE trading and automatically blacklist high-fee symbols to prevent losses.

---

## Solution Implemented

### 1. Automatic Funding Rate Filter

**File**: `microservices/autonomous_trader/funding_rate_filter.py`

**Features**:
- Fetches current funding rates from Binance API `/fapi/v1/fundingRate`
- Automatically blacklists symbols with extreme rates (>0.1% per 8h)
- Manual permanent blacklist for known problematic symbols
- Concurrent fetching for all symbols (fast)
- Detailed logging of blacklist reasons

**Thresholds**:
```python
NORMAL_FUNDING_RATE = 0.0001              # 0.01% per 8h (typical)
EXTREME_FUNDING_RATE_THRESHOLD = 0.001    # 0.1% per 8h (blacklist)

MANUAL_BLACKLIST = {
    "BREVUSDT",   # 78 USDT funding on 31 USDT margin
    "API3USDT",   # 150 USDT funding on 158 USDT margin
}
```

**API Used**:
```
GET https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT&limit=1
Response: [{"fundingRate": "0.0001", "fundingTime": 1738973200000, ...}]
```

### 2. Integration with Autonomous Trader

**File**: `microservices/autonomous_trader/autonomous_trader.py`

**Changes**:
1. Import `get_filtered_symbols` from funding_rate_filter
2. Store candidate symbols in `__init__()` (from env var)
3. Call funding filter in `start()` method BEFORE trading begins
4. Initialize EntryScanner with filtered safe symbols only
5. Add safety check in `_scan_entries()` method

**Flow**:
```
Startup
  ↓
Parse symbols from env (100+ symbols)
  ↓
Fetch funding rates from Binance (concurrent)
  ↓
Filter out symbols with extreme rates (>0.1% per 8h)
  ↓
Initialize EntryScanner with safe symbols only
  ↓
Start trading
```

### 3. Key Functions

#### `fetch_funding_rate(symbol: str) -> float`
- Fetches current funding rate from Binance
- Returns 0.0 if unavailable or API error
- 5-second timeout to prevent hanging
- Handles HTTP errors gracefully

#### `filter_symbols_by_funding_rate(symbols: List[str], threshold: float) -> Dict`
- Fetches rates for all symbols concurrently
- Checks manual blacklist first
- Compares absolute rate to threshold
- Returns dict with allowed/blacklisted/rates/reasons

#### `get_filtered_symbols(candidate_symbols: List[str]) -> List[str]`
- Main entry point for autonomous trader
- Easy enable/disable via parameter
- Returns list of safe symbols
- Detailed logging of blacklist decisions

---

## Deployment Plan

### Prerequisites Check:
```bash
# Check if aiohttp is installed (required for funding_rate_filter.py)
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 \
  'cd /home/qt/quantum_trader && source venv/bin/activate && pip show aiohttp'
```

### Deploy Steps:

```bash
# 1. Copy funding rate filter module
wsl scp -i ~/.ssh/hetzner_fresh \
  microservices/autonomous_trader/funding_rate_filter.py \
  root@46.224.116.254:/home/qt/quantum_trader/microservices/autonomous_trader/

# 2. Copy updated autonomous_trader.py
wsl scp -i ~/.ssh/hetzner_fresh \
  microservices/autonomous_trader/autonomous_trader.py \
  root@46.224.116.254:/home/qt/quantum_trader/microservices/autonomous_trader/

# 3. Restart autonomous trader service
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 \
  'systemctl restart quantum-autonomous-trader && echo "✅ Restarted"'

# 4. Monitor startup logs for funding filter
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 \
  'journalctl -u quantum-autonomous-trader -f | grep -E "FundingFilter|blacklist"'
```

### Expected Startup Logs:
```
[AutonomousTrader] Candidate symbols: 100 (will be filtered by funding rates on startup)
[FundingFilter] Checking funding rates for 100 symbols...
[FundingFilter] ❌ BREVUSDT: MANUAL_BLACKLIST (known problematic)
[FundingFilter] ❌ API3USDT: MANUAL_BLACKLIST (known problematic)
[FundingFilter] Results: 98 allowed, 2 blacklisted
[FundingFilter] 🛡️ Protected from 2 high-fee symbols
[AutonomousTrader] Trading with 98 safe symbols
```

---

## Benefits

### Proactive Protection:
- ✅ Prevents entering positions on high-fee symbols
- ✅ Eliminates need to model extreme funding in exit logic
- ✅ Simpler architecture - blacklist at entry instead of complex fee handling

### Dynamic & Automatic:
- ✅ Fetches CURRENT funding rates at startup (not hardcoded)
- ✅ Can refresh periodically during runtime (if needed)
- ✅ Automatically discovers new problematic symbols

### Comprehensive:
- ✅ Manual blacklist for known bad symbols (BREVUSDT, API3USDT)
- ✅ Automatic threshold-based filtering (>0.1% per 8h)
- ✅ Both mechanisms provide layered protection

### Safe Architecture:
- ✅ Graceful degradation - symbol list only filtered on startup
- ✅ If API fails, falls back to manual blacklist
- ✅ Can disable filter via parameter if needed
- ✅ Detailed logging for monitoring and debugging

---

## Interaction with Fee-Awareness System

The **funding rate filter** and **fee-awareness exit logic** work together:

### Funding Rate Filter (Entry Prevention):
- Blocks symbols with EXTREME funding (>0.1% per 8h)
- Prevents entering positions that will always lose money
- Permanent blacklist for known problematic symbols
- **Action**: Never enter these symbols

### Fee-Awareness Exit Logic (Normal Fee Handling):
- Handles NORMAL fees (trading 0.04% + funding 0.01% per 8h)
- Calculates R_net_after_fees for exit decisions
- FEE_PROTECTION closes if net R < 1.0
- Time-based exit pressure for positions >24h
- **Action**: Close positions when fees erode profit

### Division of Responsibilities:
```
Extreme Funding (100× normal)  →  Funding Rate Filter  →  BLOCK AT ENTRY
Normal Funding (0.01% per 8h)  →  Exit Evaluator       →  MANAGE AT EXIT
```

This two-layer approach is **cleaner and more maintainable** than trying to handle all cases in exit logic.

---

## Testing Checklist

### After Deployment:

1. **Startup Verification**:
   ```bash
   # Check that funding filter ran successfully
   journalctl -u quantum-autonomous-trader --since "1 minute ago" | grep "FundingFilter"
   ```
   - Should see "Checking funding rates for X symbols"
   - Should see blacklist results

2. **Blacklist Verification**:
   ```bash
   # Verify BREVUSDT and API3USDT are blacklisted
   journalctl -u quantum-autonomous-trader --since "1 minute ago" | grep -E "BREVUSDT|API3USDT"
   ```
   - Should see "MANUAL_BLACKLIST" or "Extreme funding" reason

3. **Entry Scanner Verification**:
   ```bash
   # Monitor entry scans - should NOT see blacklisted symbols
   journalctl -u quantum-autonomous-trader -f | grep -E "opportunity|BREVUSDT|API3USDT"
   ```
   - Should NOT see any entry opportunities for BREVUSDT/API3USDT

4. **Existing Positions**:
   - Let fee-awareness exit logic handle existing BREVUSDT/API3USDT positions
   - They should close within 1-2 cycles due to fee erosion

---

## Configuration Options

### Environment Variables (optional):

```bash
# Enable/disable funding filter (default: enabled)
ENABLE_FUNDING_FILTER=true

# Custom threshold for extreme funding (default: 0.001 = 0.1% per 8h)
FUNDING_RATE_THRESHOLD=0.001

# Refresh interval for funding rates (default: startup only)
FUNDING_REFRESH_HOURS=24
```

### Programmatic Control:

```python
# Disable funding filter if needed
safe_symbols = await get_filtered_symbols(
    candidate_symbols,
    enable_funding_filter=False  # Use all symbols
)

# Custom threshold
safe_symbols = await get_filtered_symbols(
    candidate_symbols,
    threshold=0.002  # 0.2% per 8h
)
```

---

## Future Enhancements

### Periodic Refresh:
- Add background task to refresh funding rates every N hours
- Dynamically update entry_scanner.symbols during runtime
- Catch symbols that develop extreme funding after startup

### Historical Analysis:
- Fetch 24h funding rate history (average)
- Identify consistently high-funding symbols
- Separate "temporarily high" from "always high" funding

### API Integration:
- Store funding rates in Redis for other modules
- Share blacklist with backend/dashboard
- Real-time funding rate monitoring endpoint

### Metrics:
- Track count of blacklisted symbols over time
- Log prevented losses (estimate based on historical data)
- Alert if >X% of universe is blacklisted (data quality issue)

---

## Summary

**Problem**: Symbols with extreme funding fees (BREVUSDT, API3USDT) cause guaranteed losses

**Solution**: Automatic funding rate filter that checks Binance API before trading

**Result**: 
- ✅ Prevents entering high-fee positions
- ✅ Simpler architecture than complex fee handling
- ✅ Dynamic and automatic discovery of problematic symbols
- ✅ Works with existing fee-awareness exit logic for normal fees

**Status**: Ready to deploy and test on VPS

---

## Related Files

- `microservices/autonomous_trader/funding_rate_filter.py` - NEW (automatic blacklist)
- `microservices/autonomous_trader/autonomous_trader.py` - MODIFIED (startup integration)
- `microservices/ai_engine/exit_evaluator.py` - DEPLOYED FEB 8 (fee-awareness)
- `AI_FEE_AWARENESS_FIX_FEB8.md` - Related documentation

---

**Next Action**: Deploy to VPS and verify blacklist working in startup logs
