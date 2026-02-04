# P3 Harvest API Fallback & Backfill SUCCESS

**Date:** 2026-02-04 02:07 UTC  
**Commits:** 61dff1c71, aa6fe6b3a, 238096909, b5839bff6  
**Status:** ✅ API fallback deployed, ⚠️ entry_price backfill blocked by env config

---

## 1. Problem Analysis

### Initial Symptoms
- HarvestBrain deployed (HARVEST_MODE=live), policy configured (min_r=0.5, ladder=0.5:0.25,1.0:0.25,1.5:0.25)
- `HARVEST_TICK scanned=44 evaluated=0 emitted=0 skipped=1`
- **0 positions evaluated despite 44 open positions**

### Root Cause #1: Missing Mark Price Data
```bash
# Diagnosis
redis-cli KEYS "quantum:ticker:*"   # → (empty array)
redis-cli KEYS "quantum:market:*"   # → (empty array)
```

**Explanation:**  
- `quantum-marketstate.service` is running but NOT populating ticker/market keys
- `_get_mark_price()` returned 0 for all symbols
- All positions skipped with `SKIP_NO_MARK_PRICE` (logged at DEBUG level, not visible)

### Root Cause #2: Missing Risk Fields
```python
# Position hash inspection
quantum:position:ADAUSDT
- atr_value: 0 (MISSING)
- volatility_factor: 0 (MISSING)
- entry_risk_usdt: 0 (MISSING)
```

**Explanation:**  
- Positions created BEFORE apply_layer patch (commit 1c713518b)
- Apply-layer only stores risk fields for NEW ENTRY orders
- 43/44 positions lack atr/vol/entry_risk → R_net cannot be computed

### Root Cause #3: Missing Entry Price
```bash
# Final blocker discovered
ADAUSDT: qty=666.22 entry=MISSING
ANKRUSDT: qty=33892.56 entry=MISSING
...
Summary: 1 have entry_price, 43 missing entry_price
```

**Explanation:**  
- `apply_layer/main.py` places ENTRY orders but does NOT parse `avgPrice` from Binance API response
- BinanceTestnetClient.place_market_order() returns: `{orderId, symbol, side, quantity, executedQty, status}`
- **avgPrice NOT extracted**, position hash stores `entry_price=0`
- HarvestBrain skips positions where `entry_price == 0` (line 1156)

---

## 2. Solutions Implemented

### 2.1 API Fallback for Mark Price (✅ DEPLOYED)
**Commit:** 61dff1c71

**Changes:**
```python
# harvest_brain.py: _get_mark_price() now returns (price, source)
async def _get_mark_price(self, symbol: str) -> tuple[float, str]:
    # 1. Check in-memory cache (2s TTL)
    if symbol in self.price_cache:
        ...
        return (price, 'cache')
    
    # 2. Try quantum:ticker:{symbol}
    ticker_data = self.redis.hgetall(f"quantum:ticker:{symbol}")
    if ticker_data and ticker_data.get('markPrice'):
        return (float(ticker_data['markPrice']), 'redis_ticker')
    
    # 3. Try quantum:market:{symbol}
    market_data = self.redis.hgetall(f"quantum:market:{symbol}")
    if market_data and market_data.get('price'):
        return (float(market_data['price']), 'redis_market')
    
    # 4. API fallback (rate-limited: max 10 calls per tick)
    if self.api_calls_this_tick >= self.max_api_calls_per_tick:
        return (0.0, 'rate_limited')
    
    url = f"https://testnet.binancefuture.com/fapi/v1/ticker/price?symbol={symbol}"
    response = urllib.request.urlopen(url, timeout=2)
    data = json.loads(response.read().decode())
    price = float(data['price'])
    
    self.api_calls_this_tick += 1
    self.price_cache[symbol] = (price, time.time())
    return (price, 'api')
```

**Features:**
- In-memory cache with 2s TTL (reduces API calls for repeated evaluations)
- Rate limiting: max 10 API calls per tick (5s scan interval)
- Graceful degradation: redis_ticker → redis_market → api → rate_limited → unavailable
- `mark_source` logged in `HARVEST_EVAL` for observability

**Proof:**
```log
2026-02-04 01:56:53,413 | INFO | HARVEST_EVAL symbol=ARCUSDT mark=0.074980 entry=0.063600 pnl=35.6852 risk=1567.8896 R_net=0.023 mark_source=cache
2026-02-04 01:56:58,828 | INFO | HARVEST_EVAL symbol=ARCUSDT mark=0.074940 entry=0.063600 pnl=35.6852 risk=1567.8896 R_net=0.023 mark_source=api
2026-02-04 01:56:58,834 | INFO | HARVEST_TICK scanned=44 evaluated=1 emitted=0 skipped_risk=0 skipped_price=0 api_calls=1
```

### 2.2 Backfill Risk Fields from trade.intent (✅ COMPLETE)
**Commit:** 238096909  
**Script:** `scripts/backfill_position_risk.py`

**Approach:**
```python
# Find last signal for each symbol in trade.intent stream
entries = redis.xrevrange("quantum:stream:trade.intent", count=5000)
for entry_id, fields in entries:
    if fields['event_type'] == 'trade.intent':
        payload = json.loads(fields['payload'])
        if payload['symbol'] == target_symbol:
            atr_value = payload['atr_value']
            volatility_factor = payload['volatility_factor']
            # Compute and store
            risk_price = atr_value * volatility_factor
            entry_risk_usdt = abs(qty) * risk_price
            redis.hset(pos_key, mapping={
                'atr_value': atr_value,
                'volatility_factor': volatility_factor,
                'risk_price': risk_price,
                'entry_risk_usdt': entry_risk_usdt,
                'risk_missing': 0
            })
```

**Execution:**
```bash
ssh root@46.224.116.254 'python scripts/backfill_position_risk.py'
```

**Results:**
```
[OK] ADAUSDT: atr=0.0398 vol=1.0 entry_risk=26.80
[OK] ANKRUSDT: atr=0.1000 vol=2.4 entry_risk=8199.60
[OK] CHESSUSDT: atr=0.1000 vol=5.0 entry_risk=4264.39
...
[SKIP] ARCUSDT: Already has entry_risk_usdt=1567.89

Backfill Complete:
  Updated: 43
  Skipped (already valid): 1
  Failed (no intent found): 0
  Total processed: 44
```

### 2.3 Backfill Entry Price from Binance API (⚠️ BLOCKED)
**Commit:** b5839bff6

**Implementation:**
```python
async def _fetch_exchange_positions(self) -> None:
    """Fetch all positions from Binance /fapi/v2/positionRisk (30s cache)"""
    # Call Binance API with signed request
    url = f"https://testnet.binancefuture.com/fapi/v2/positionRisk?timestamp={ts}&signature={sig}"
    response = urllib.request.urlopen(url, timeout=5)
    positions = json.loads(response.read().decode())
    
    # Cache open positions
    for pos in positions:
        if float(pos['positionAmt']) != 0:
            self.exchange_positions_cache[pos['symbol']] = {
                'entryPrice': float(pos['entryPrice']),
                'positionAmt': float(pos['positionAmt']),
                'unrealizedProfit': float(pos['unRealizedProfit'])
            }

async def _enrich_position_from_redis(self, symbol: str) -> None:
    ...
    # Backfill entry_price from exchange if missing
    if qty != 0 and entry_price == 0:
        if symbol in self.exchange_positions_cache:
            exchange_entry = self.exchange_positions_cache[symbol]['entryPrice']
            if exchange_entry > 0:
                entry_price = exchange_entry
                redis.hset(pos_key, 'entry_price', str(entry_price))
                logger.info(f"[BACKFILL] {symbol}: entry_price={entry_price} from exchange API")
```

**Blocker:**
```bash
# Diagnosis
systemctl status quantum-harvest-brain.service  # → active (running)
cat /etc/systemd/system/quantum-harvest-brain.service
# → EnvironmentFile=/etc/quantum/harvest-brain.env

cat /etc/quantum/harvest-brain.env | grep BINANCE
# → BINANCE_TESTNET_API_KEY=...
# → BINANCE_TESTNET_API_SECRET=...

# BUT: env vars not loading into process
# Suspect: Windows line endings (\r\n) in env file
```

**Attempted Fix:**
```bash
sed -i 's/\r$//' /etc/quantum/harvest-brain.env
systemctl daemon-reload
systemctl restart quantum-harvest-brain.service
```

**Status:** Still blocked - no `BACKFILL` or `Fetched.*exchange` messages in logs

---

## 3. Current State

### What Works ✅
1. **API Fallback for Prices**  
   - `evaluated=1` (ARCUSDT, manually patched with entry_price)
   - `mark_source=api`, `mark_source=cache` confirmed
   - `api_calls=1` per tick (rate limiting functional)

2. **Risk Fields Backfilled**  
   - 43/44 positions now have `atr_value`, `volatility_factor`, `entry_risk_usdt`
   - Example: `ADAUSDT: atr=0.0398 vol=1.0 entry_risk=26.80`

3. **Periodic Scanning**  
   - `HARVEST_TICK scanned=44` every 5s
   - Non-blocking apply.result consumer (100ms block time)

### What's Blocked ⚠️
1. **43 Positions Cannot Be Evaluated**  
   - Blocker: `entry_price == 0`  
   - Code: `if qty == 0 or entry_price == 0: return` (line 1156)
   - Binance API backfill implemented but env vars not loading in systemd service

2. **Evaluated Count Stuck at 1**  
   - Only ARCUSDT (manually patched) is being evaluated
   - All other positions skip silently (no SKIP log because rate-limited to 60s)

---

## 4. Recommended Fix: Update apply_layer

**Problem:** `apply_layer/main.py` does NOT store actual fill price after ENTRY orders.

**Current Code:**
```python
# Line 2335-2345: apply_layer places ENTRY order
order_result = client.place_market_order(
    symbol=symbol,
    side=side,
    quantity=qty,
    reduce_only=False
)

logger.info(f"[ENTRY] {symbol}: {side} order placed: {order_result}")
# order_result contains: {orderId, symbol, side, quantity, executedQty, status}
# avgPrice NOT extracted!
```

**Proposed Fix:**
```python
# Update BinanceTestnetClient.place_market_order() to return avgPrice
def place_market_order(self, symbol, side, quantity, reduce_only=False):
    ...
    result = self._request('POST', '/fapi/v1/order', params=params, signed=True)
    return {
        'orderId': result.get('orderId'),
        'symbol': result.get('symbol'),
        'side': result.get('side'),
        'quantity': result.get('origQty'),
        'executedQty': result.get('executedQty'),
        'avgPrice': result.get('avgPrice'),  # ← ADD THIS
        'status': result.get('status'),
        'reduceOnly': reduce_only
    }

# Line 2363: Use avgPrice as entry_price
entry_price_actual = float(order_result.get('avgPrice', entry_price))
position_mapping = {
    ...
    "entry_price": str(entry_price_actual),  # ← Use actual fill price
    ...
}
```

**Alternative:** Parse `FILL` order update from websocket if available.

---

## 5. Operational Checklist

### Immediate Actions
- [ ] Fix env var loading in harvest-brain.service OR
- [ ] Update apply_layer to store avgPrice (recommended)
- [ ] Verify 44 positions get entry_price (currently 1/44)
- [ ] Confirm `evaluated=44` in HARVEST_TICK logs
- [ ] Monitor for HARVEST_EMIT actions with R_net > 0.5

### Monitoring
```bash
# Check evaluated count
ssh root@46.224.116.254 'tail -20 /tmp/harvest_periodic.log | grep HARVEST_TICK'

# Verify entry_price backfill
ssh root@46.224.116.254 'redis-cli HGET quantum:position:ADAUSDT entry_price'

# Count positions with entry_price
ssh root@46.224.116.254 'python3 -c "
import redis
r = redis.Redis(decode_responses=True)
cursor, keys = 0, []
while True:
    cursor, k = r.scan(cursor=cursor, match=\"quantum:position:*\", count=100)
    keys.extend([x for x in k if \":ledger:\" not in x and \":snapshot:\" not in x])
    if cursor == 0: break

has_entry = sum(1 for k in keys if float(r.hget(k, \"entry_price\") or 0) > 0)
print(f\"{has_entry}/{len(keys)} have entry_price\")
"'
```

### Verification Commands
```bash
# Proof: API fallback works
ssh root@46.224.116.254 'timeout 10 /opt/quantum/venvs/ai-client-base/bin/python -u \
  /home/qt/quantum_trader/microservices/harvest_brain/harvest_brain.py 2>&1 | \
  grep -E "(mark_source=api|api_calls=)"'

# Proof: Risk fields backfilled
ssh root@46.224.116.254 'redis-cli HGETALL quantum:position:ADAUSDT | \
  grep -E "(atr_value|volatility_factor|entry_risk_usdt)"'

# Proof: Positions still missing entry_price
ssh root@46.224.116.254 'redis-cli HGET quantum:position:ADAUSDT entry_price'
# Expected: (nil) or 0
```

---

## 6. Technical Debt

1. **apply_layer MUST store avgPrice**  
   - Current: Uses plan entry_price (pre-execution estimate)
   - Fix: Parse avgPrice from order response, store in position hash

2. **marketstate service not populating ticker keys**  
   - quantum:ticker:* and quantum:market:* are empty
   - API fallback is working but adds latency
   - Should fix underlying price feed service

3. **Systemd EnvironmentFile loading issues**  
   - Env vars added to /etc/quantum/harvest-brain.env
   - May need explicit `Environment=` directives in service file instead

---

## 7. Commits

| Commit | Description |
|--------|-------------|
| 61dff1c71 | P3 Harvest: Add API fallback for mark price + backfill script |
| aa6fe6b3a | Fix _get_mark_price tuple unpack in _enrich_position_from_redis |
| 238096909 | Fix backfill script to parse trade.intent signal payloads |
| b5839bff6 | P3 Harvest: Backfill entry_price from Binance API |

**Total Lines Changed:** +265 harvest_brain.py, +183 backfill_position_risk.py

---

## 8. Next Steps

**Option A: Quick Fix (Recommended)**
1. Update apply_layer to store avgPrice from order response
2. Restart apply_layer service
3. Wait for next ENTRY order to test
4. Run backfill script to update existing 43 positions from Binance API manually

**Option B: Debug Env Loading**
1. Add explicit `Environment=BINANCE_TESTNET_API_KEY=...` to systemd service file
2. Restart harvest-brain service
3. Verify backfill executes automatically

**Option C: Manual Patch (Immediate)**
1. Write Python script to fetch all positions from Binance API
2. Update Redis `quantum:position:*` entry_price fields directly
3. Verify `evaluated=44` in logs

---

**Report Generated:** 2026-02-04 02:07 UTC  
**System:** VPS 46.224.116.254 (quantumtrader-prod-1)  
**Environment:** Python 3.12, Redis 6.x, Binance Testnet Futures
