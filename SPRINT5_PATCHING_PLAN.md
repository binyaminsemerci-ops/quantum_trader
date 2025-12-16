# Sprint 5 Del 3: Patching Plan - Top 10 Critical Gaps

**Status**: 🔧 **ACTIVE** - Patching known issues without stress tests  
**Strategy**: Minimal, safe patches for production readiness

---

## Patch Priority Matrix

| # | Gap | Priority | Complexity | Est. Time | Status |
|---|-----|----------|------------|-----------|--------|
| 1 | Redis Outage Handling | P0-CRITICAL | HIGH | 45 min | 🔲 TODO |
| 2 | Binance Rate Limiting | P0-CRITICAL | MEDIUM | 30 min | 🔲 TODO |
| 3 | Signal Flood Throttling | P0-CRITICAL | MEDIUM | 30 min | 🔲 TODO |
| 4 | AI Engine Mock Data | P0-CRITICAL | LOW | 20 min | 🔲 TODO |
| 5 | Portfolio PnL Consistency | P0-CRITICAL | MEDIUM | 30 min | 🔲 TODO |
| 6 | WS Dashboard Load | P0-CRITICAL | MEDIUM | 30 min | 🔲 TODO |
| 7 | ESS Reset Logic | P1-HIGH | LOW | 15 min | 🔲 TODO |
| 8 | PolicyStore Aging | P1-HIGH | LOW | 15 min | 🔲 TODO |
| 9 | Execution Retry Policy | P1-HIGH | MEDIUM | 25 min | 🔲 TODO |
| 10 | Health Monitoring Service | P2-MEDIUM | HIGH | 60 min | 🔲 TODO |

**Total Estimated Time**: ~4.5 hours

---

## Patch #1: Redis Outage Handling (P0-CRITICAL)

### Problem
- **Issue**: No disk-buffer fallback when Redis down 60-120 seconds
- **Impact**: All events lost, system cannot resync after Redis recovery
- **Location**: `backend/domains/architecture_v2/event_bus.py`

### Solution
**Implement SQLite-based circular buffer for event buffering**

**Files to Modify**:
- `backend/domains/architecture_v2/event_bus.py`

**Implementation**:
```python
# Add to EventBusV2 class:

class EventBusV2:
    def __init__(self, redis_url: str, fallback_db_path: str = "/app/data/eventbus_fallback.db"):
        self.redis_url = redis_url
        self.redis_client = None
        self.fallback_db_path = fallback_db_path
        self.use_fallback = False
        self._init_fallback_db()
    
    def _init_fallback_db(self):
        """Initialize SQLite fallback database"""
        import sqlite3
        conn = sqlite3.connect(self.fallback_db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS event_buffer (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                payload TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON event_buffer(timestamp)")
        conn.commit()
        conn.close()
    
    async def publish_event(self, event_type: str, payload: dict):
        """Publish event with fallback to SQLite"""
        try:
            # Try Redis first
            if self.redis_client and not self.use_fallback:
                await self._publish_to_redis(event_type, payload)
            else:
                # Use SQLite fallback
                await self._publish_to_fallback(event_type, payload)
        except Exception as e:
            logger.warning(f"Redis publish failed, switching to fallback: {e}")
            self.use_fallback = True
            await self._publish_to_fallback(event_type, payload)
    
    async def _publish_to_fallback(self, event_type: str, payload: dict):
        """Publish to SQLite fallback"""
        import sqlite3
        import json
        import time
        
        conn = sqlite3.connect(self.fallback_db_path)
        conn.execute(
            "INSERT INTO event_buffer (event_type, payload, timestamp) VALUES (?, ?, ?)",
            (event_type, json.dumps(payload), time.time())
        )
        conn.commit()
        
        # Cleanup old events (keep last 10000)
        conn.execute("""
            DELETE FROM event_buffer 
            WHERE id NOT IN (
                SELECT id FROM event_buffer ORDER BY id DESC LIMIT 10000
            )
        """)
        conn.commit()
        conn.close()
    
    async def resync_from_fallback(self):
        """Resync buffered events to Redis after recovery"""
        import sqlite3
        import json
        
        if not self.use_fallback:
            return
        
        conn = sqlite3.connect(self.fallback_db_path)
        cursor = conn.execute("SELECT event_type, payload FROM event_buffer ORDER BY id ASC")
        
        resynced = 0
        for row in cursor:
            event_type, payload_json = row
            payload = json.loads(payload_json)
            try:
                await self._publish_to_redis(event_type, payload)
                resynced += 1
            except Exception as e:
                logger.error(f"Resync failed for event {event_type}: {e}")
                break
        
        if resynced > 0:
            # Clear synced events
            conn.execute("DELETE FROM event_buffer")
            conn.commit()
            logger.info(f"✅ Resynced {resynced} events from fallback to Redis")
        
        conn.close()
        self.use_fallback = False
```

**Testing**:
- Unit test: Simulate Redis down, verify SQLite buffering
- Integration test: Redis down → buffer → Redis up → resync

---

## Patch #2: Binance Rate Limiting (P0-CRITICAL)

### Problem
- **Issue**: Basic retry, no exponential backoff for API errors -1003/-1015
- **Impact**: Risk of account ban from sustained API abuse
- **Location**: `microservices/execution/service.py`, `backend/services/execution/binance_adapter.py`

### Solution
**Implement token bucket + exponential backoff**

**Files to Modify**:
- `backend/services/execution/binance_adapter.py`

**Implementation**:
```python
import time
from collections import deque

class BinanceRateLimiter:
    """Token bucket rate limiter with exponential backoff"""
    
    def __init__(self, max_requests: int = 1200, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = deque()
        self.backoff_until = 0
        self.backoff_duration = 1.0  # Start with 1 second
    
    def can_request(self) -> bool:
        """Check if request is allowed"""
        now = time.time()
        
        # Check if in backoff period
        if now < self.backoff_until:
            return False
        
        # Remove old requests outside window
        while self.requests and self.requests[0] < now - self.window_seconds:
            self.requests.popleft()
        
        # Check if under limit
        return len(self.requests) < self.max_requests
    
    def record_request(self):
        """Record a successful request"""
        self.requests.append(time.time())
        self.backoff_duration = 1.0  # Reset backoff on success
    
    def record_rate_limit_error(self):
        """Record rate limit error and trigger exponential backoff"""
        self.backoff_duration = min(self.backoff_duration * 2, 300)  # Max 5 min
        self.backoff_until = time.time() + self.backoff_duration
        logger.warning(f"⚠️ Binance rate limit hit, backing off for {self.backoff_duration}s")

# Add to BinanceAdapter class:
class BinanceAdapter:
    def __init__(self):
        self.rate_limiter = BinanceRateLimiter(max_requests=1200, window_seconds=60)
    
    async def execute_order(self, symbol: str, side: str, quantity: float, **kwargs):
        # Wait if rate limited
        while not self.rate_limiter.can_request():
            wait_time = self.rate_limiter.backoff_until - time.time()
            logger.info(f"⏳ Rate limited, waiting {wait_time:.1f}s...")
            await asyncio.sleep(min(wait_time, 1.0))
        
        try:
            result = await self._place_order(symbol, side, quantity, **kwargs)
            self.rate_limiter.record_request()
            return result
        except BinanceAPIException as e:
            if e.code in [-1003, -1015]:  # Rate limit errors
                self.rate_limiter.record_rate_limit_error()
                raise
```

**Testing**:
- Unit test: Simulate rate limit errors, verify backoff
- Stress test: 2000 requests, verify no ban

---

## Patch #3: Signal Flood Throttling (P0-CRITICAL)

### Problem
- **Issue**: AI Engine can generate 30-50 signals/sec, no throttling in Execution
- **Impact**: Execution overload, order queue backup, missed trades
- **Location**: `microservices/execution/service.py`

### Solution
**Add queue size limit + signal dropping policy**

**Files to Modify**:
- `backend/services/execution/event_driven_executor.py`

**Implementation**:
```python
from collections import deque

class EventDrivenExecutor:
    def __init__(self, max_queue_size: int = 100):
        self.signal_queue = deque(maxlen=max_queue_size)
        self.dropped_signals = 0
    
    async def handle_signal(self, signal: dict):
        """Handle incoming signal with queue limit"""
        if len(self.signal_queue) >= self.signal_queue.maxlen:
            self.dropped_signals += 1
            logger.warning(f"⚠️ Signal queue full ({self.signal_queue.maxlen}), dropping signal for {signal.get('symbol')}")
            
            # Drop lowest confidence signal
            if signal.get('confidence', 0) > min(s.get('confidence', 0) for s in self.signal_queue):
                # Replace lowest confidence with new signal
                min_signal = min(self.signal_queue, key=lambda s: s.get('confidence', 0))
                self.signal_queue.remove(min_signal)
                self.signal_queue.append(signal)
            return
        
        self.signal_queue.append(signal)
    
    async def process_queue(self):
        """Process signals from queue with rate limiting"""
        while self.signal_queue:
            signal = self.signal_queue.popleft()
            
            try:
                await self._execute_signal(signal)
            except Exception as e:
                logger.error(f"Signal execution failed: {e}")
            
            # Rate limit: max 2 signals/second
            await asyncio.sleep(0.5)
```

**Testing**:
- Stress test: Generate 500 signals in 10 seconds
- Verify: Queue doesn't exceed 100, lowest confidence dropped

---

## Patch #4: AI Engine Mock Data (P0-CRITICAL)

### Problem
- **Issue**: Metrics endpoints return mock data instead of real ML model outputs
- **Impact**: Dashboard shows fake strategy insights
- **Location**: `microservices/ai_engine/api.py`, `backend/api/dashboard/routes.py`

### Solution
**Connect metrics endpoints to real ML models**

**Files to Modify**:
- `backend/api/dashboard/routes.py`

**Implementation**:
```python
@router.get("/api/dashboard/ml_metrics")
async def get_ml_metrics():
    """Get real ML model metrics"""
    try:
        from backend.services.ai.ai_trading_engine import AITradingEngine
        
        ai_engine = get_ai_engine()  # Get singleton instance
        
        if not ai_engine:
            return {"error": "AI Engine not initialized"}
        
        # Get real metrics from ensemble
        ensemble_metrics = {
            "xgb_accuracy": ai_engine.ensemble.xgb_model.get_accuracy() if hasattr(ai_engine.ensemble, 'xgb_model') else 0.0,
            "lgbm_accuracy": ai_engine.ensemble.lgbm_model.get_accuracy() if hasattr(ai_engine.ensemble, 'lgbm_model') else 0.0,
            "tft_accuracy": ai_engine.ensemble.tft_model.get_accuracy() if hasattr(ai_engine.ensemble, 'tft_model') else 0.0,
            "ensemble_confidence": ai_engine.last_signal_confidence if hasattr(ai_engine, 'last_signal_confidence') else 0.0,
        }
        
        return {
            "status": "ok",
            "models": ensemble_metrics,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get ML metrics: {e}")
        return {"error": str(e)}
```

**Testing**:
- Integration test: Call endpoint, verify non-mock data
- Dashboard: Verify real metrics displayed

---

## Remaining Patches (5-10)

Will implement in next batch after testing patches 1-4.

---

## Testing Strategy

1. **Unit Tests**: For each patch, create focused test
2. **Integration Tests**: Test patches together
3. **Manual Verification**: Check logs, metrics, behavior
4. **Rollback Plan**: Keep original code in comments for quick revert

---

## Patch Status Tracking

Format: `Issue → Fix → File/Lines → Status`

Will update as patches are implemented.
