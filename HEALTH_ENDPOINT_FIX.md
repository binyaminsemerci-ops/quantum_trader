# Health Endpoint Timeout Fix

## Problem
Health endpoints (`/health`, `/api/v2/health`) timing out after 5+ seconds, preventing monitoring and Docker health checks.

## Root Cause
`backend/core/health.py` makes synchronous network calls to Binance Testnet with 5-second timeout:
```python
async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
```

When Binance Testnet is slow or unresponsive, this blocks the entire health check.

## Solution
Reduce Binance health check timeout to **1 second** and add graceful degradation:

### File: `backend/core/health.py`

**Line 376: Change timeout from 5s → 1s**
```python
# BEFORE:
async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:

# AFTER:
async with session.get(url, timeout=aiohttp.ClientTimeout(total=1)) as resp:
```

**Line 392: Update error message**
```python
# BEFORE:
error="Timeout after 5s",

# AFTER:
error="Timeout after 1s",
```

**Line 424: Change WebSocket timeout from 5s → 1s**
```python
# BEFORE:
timeout=aiohttp.ClientTimeout(total=5)

# AFTER:
timeout=aiohttp.ClientTimeout(total=1)
```

**Line 441: Update error message**
```python
# BEFORE:
error="Timeout after 5s",

# AFTER:
error="Timeout after 1s",
```

### Alternative: Skip Binance Check for Basic Health

Add a **lightweight `/health/live`** endpoint that only checks process liveness:

**File: `backend/main.py`** (add new endpoint):
```python
@app.get("/health/live", tags=["Health"])
async def health_liveness():
    """
    Lightweight liveness check - just confirms process is alive.
    Does not check dependencies. Use for Docker/K8s liveness probe.
    """
    return {
        "status": "ok",
        "service": "quantum_trader",
        "timestamp": datetime.utcnow().isoformat()
    }
```

**File: `docker-compose.yml`** (update healthcheck):
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health/live"]
  interval: 10s
  timeout: 2s
  retries: 3
  start_period: 30s
```

## Priority
🔴 **CRITICAL (P0)** - Blocks system monitoring and Docker health checks

## Testing After Fix
```powershell
# Test fast liveness check
Invoke-RestMethod http://localhost:8000/health/live  # Should return in <100ms

# Test full health check (may take up to 1s for Binance)
Invoke-RestMethod http://localhost:8000/api/v2/health  # Should return in <2s

# Test Docker healthcheck
docker ps  # Backend should show (healthy) status after 30s
```

## Impact
- Health endpoints respond in <2 seconds (down from 5+)
- Docker containers show proper health status
- Monitoring systems can track service health
- Slow Binance Testnet degrades gracefully (marks dependency CRITICAL but returns quickly)

## Applied
✅ Fix documented, ready to apply
⏳ Waiting for user confirmation to modify files
