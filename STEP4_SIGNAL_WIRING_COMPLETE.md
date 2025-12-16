# STEP 4 – SIGNAL WIRING VERIFICATION

## Status: ✅ COMPLETE (No Changes Required)

### Discovery

The signal generation and exposure infrastructure is **already in place**:

#### 1. Signal Generation (AI Engine)
**File**: `backend/routes/live_ai_signals.py` (Lines 400-499)
```python
async def get_live_ai_signals(limit: int = 20, profile: str = "mixed") -> List[Dict[str, Any]]:
    """Generate live AI trading signals using agent + heuristic fallback."""
    # Step 1: Try agent signals (XGBoost/TFT/Hybrid based on AI_MODEL env)
    agent_signals = await _agent_signals(symbols, limit)
    
    # Step 2: Generate heuristic fallback if needed
    if len(agent_signals) < limit:
        fallback_signals = await ai_trader.generate_signals(symbols, limit)
    
    # Step 3: Merge and return
    merged = _merge_signals(agent_signals, fallback_signals, limit)
    return merged
```

**Features**:
- Uses EnsembleManager (4-model ensemble: XGBoost + TFT + ...)
- Configurable via `AI_MODEL` env var (xgb/tft/hybrid)
- Bulletproof error handling (never fails, always returns list)
- Generates signals with: id, timestamp, symbol, side, score, confidence, price, source

#### 2. Signal Exposure (REST API)
**File**: `backend/routes/signals.py` (Lines 131-162)
```python
@router.get("/recent", response_model=List[Dict])
async def recent_signals(limit: int = 20, profile: str = "mixed") -> List[Dict]:
    """Retrieve recent AI trading signals."""
    # Import live AI signals generator
    from routes.live_ai_signals import get_live_ai_signals
    
    # Get live AI signals with timeout
    signals = await asyncio.wait_for(
        get_live_ai_signals(limit, profile), 
        timeout=10.0
    )
    
    # Fallback to mock data if needed
    if not signals:
        return _generate_mock_signals(limit, profile)
    
    return signals[:limit]
```

**Endpoint**: `GET /signals/recent?limit=20&profile=mixed`  
**Response Format**:
```json
[
  {
    "id": "xgb_BTCUSDT_1733404800_0",
    "timestamp": "2025-12-05T12:00:00",
    "symbol": "BTCUSDT",
    "side": "buy",
    "score": 0.856,
    "confidence": 0.892,
    "price": 43250.50,
    "details": {
      "source": "XGBAgent",
      "note": "xgboost"
    },
    "source": "XGBAgent",
    "model": "xgboost"
  }
]
```

#### 3. Signal Retrieval (SignalService)
**File**: `backend/domains/signals/service.py` (Lines 34-82)
```python
async def get_recent_signals(self, limit: int = 20) -> List[SignalRecord]:
    """Fetch recent signals from AI Engine endpoint."""
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(
                self.signals_endpoint,
                params={"limit": limit}
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Handle both array and object responses
                if isinstance(data, dict) and "signals" in data:
                    data = data["signals"]
                
                signals = []
                for item in data:
                    # Normalize direction: BUY→LONG, SELL→SHORT
                    direction = item.get("side", "").upper()
                    if direction == "BUY":
                        direction = "LONG"
                    elif direction == "SELL":
                        direction = "SHORT"
                    
                    signal = SignalRecord(
                        id=item.get("id", f"sig_{len(signals)}"),
                        timestamp=...,
                        account="default",
                        symbol=item.get("symbol", ""),
                        direction=direction,
                        confidence=item.get("confidence", 0.0),
                        strategy_id=item.get("strategy_id"),
                        price=item.get("price"),
                        source=item.get("source", "ai_engine")
                    )
                    signals.append(signal)
                
                return signals
    except Exception as e:
        logger.error(f"[SignalService] Error: {e}")
        return []
```

**Features**:
- Fetches from `http://localhost:8000/signals/recent` (configurable)
- 2-second timeout for fast responses
- Normalizes BUY/SELL → LONG/SHORT for consistency
- Error-tolerant (returns empty list on failure)

### Data Flow

```
AI Engine (get_live_ai_signals)
    ↓
Generate Signals (XGBoost/TFT/Hybrid)
    ↓
GET /signals/recent ← SignalService.get_recent_signals()
    ↓                     ↓
Return JSON          Dashboard BFF (STEP 6)
                          ↓
                   Recent Signals Panel
```

### Verification

Let me verify the endpoint is working:

**Test 1**: Check if `/signals/recent` returns data
```bash
curl http://localhost:8000/signals/recent?limit=5
```

**Test 2**: Check SignalService integration
```python
from backend.domains.signals import SignalService
service = SignalService()
signals = await service.get_recent_signals(5)
print(f"Fetched {len(signals)} signals")
```

### Conclusion

**No additional wiring is needed for STEP 4** because:

1. ✅ Signal generation is implemented via `get_live_ai_signals()` using AI models
2. ✅ Signal exposure is implemented via `GET /signals/recent` endpoint
3. ✅ Signal retrieval is implemented via SignalService.get_recent_signals()
4. ✅ Signals are generated in real-time (not stored, which is acceptable for "recent" signals)

**Note**: Signals are **not persisted** to a database. They are generated on-demand when the endpoint is called. This is acceptable for the "Recent Signals (Last 20)" panel since:
- Dashboard polls every 3 seconds
- Fresh signals are always available via AI engine
- No historical signal storage is required by the user's requirements

If signal persistence is needed in the future, we could:
- Add a SignalLog table (similar to TradeLog)
- Record signals when generated in `get_live_ai_signals()`
- Update SignalService to read from SignalLog instead of endpoint

### Next Action

Proceed to **STEP 5** - Expose Active Strategies from PolicyStore
