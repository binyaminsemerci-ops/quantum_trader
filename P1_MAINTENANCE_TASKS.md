# ============================================================================
# P1 Maintenance Tasks - First Maintenance Window (Est. 2-3 hours)
# ============================================================================
# Created: December 3, 2025
# Priority: P1 (Non-blocking for production launch)
# Status: Scheduled for first maintenance window
# ============================================================================

## OVERVIEW

These P1 improvements were identified during IB-C-3 consistency sweep and do NOT block production deployment. They address observability gaps, performance optimizations, and technical debt cleanup.

**Total Estimated Time:** 2-3 hours
**Recommended Schedule:** During low-traffic hours or scheduled maintenance window

---

## TASK P1-1: Add Missing Critical Events (30 minutes)

### Issue
EventBus v2 is missing 4 critical events that reduce observability:
- `ai.hfos.mode_changed` - HedgeFundOS mode transitions
- `safety.governor.level_changed` - Safety level adjustments
- `model.shadow_test.completed` - Shadow testing results
- `federation.node.status_changed` - Federation node health

### Impact
- Reduced observability of AI OS mode changes
- Manual checking required for safety level transitions
- Harder to debug Federation v2 node issues

### Implementation

#### File: `backend/core/event_bus.py`

Add event definitions to existing event map:

```python
# Existing events...
"model.promoted": [],
"model.rollback": [],

# ADD THESE:
"ai.hfos.mode_changed": [],           # HedgeFundOS mode transitions
"safety.governor.level_changed": [],  # Safety adjustments
"model.shadow_test.completed": [],    # Shadow test results
"federation.node.status_changed": [], # Federation health
```

#### File: `backend/services/ai/hfos_manager.py`

Add event publish on mode changes:

```python
async def set_mode(self, mode: str):
    old_mode = self.current_mode
    self.current_mode = mode
    
    # ADD THIS:
    await self.event_bus.publish("ai.hfos.mode_changed", {
        "old_mode": old_mode,
        "new_mode": mode,
        "timestamp": datetime.utcnow().isoformat()
    })
    
    logger.info(f"HedgeFundOS mode changed: {old_mode} → {mode}")
```

#### File: `backend/services/safety/governor.py`

Add event publish on safety level changes:

```python
async def set_safety_level(self, level: int):
    old_level = self.current_level
    self.current_level = level
    
    # ADD THIS:
    await self.event_bus.publish("safety.governor.level_changed", {
        "old_level": old_level,
        "new_level": level,
        "reason": "manual_override",
        "timestamp": datetime.utcnow().isoformat()
    })
    
    logger.info(f"Safety level changed: {old_level} → {level}")
```

#### File: `backend/federation/federated_engine_v2.py`

Add event publish on node status changes:

```python
async def _update_node_status(self, node_id: str, status: str):
    old_status = self.node_statuses.get(node_id, "unknown")
    self.node_statuses[node_id] = status
    
    # ADD THIS:
    await self.event_bus.publish("federation.node.status_changed", {
        "node_id": node_id,
        "old_status": old_status,
        "new_status": status,
        "timestamp": datetime.utcnow().isoformat()
    })
```

### Testing

```bash
# Test event publication
docker compose exec backend python -c "
from backend.core.event_bus import get_event_bus
import asyncio

async def test():
    bus = get_event_bus()
    await bus.publish('ai.hfos.mode_changed', {'test': True})
    print('✓ Event published')

asyncio.run(test())
"

# Monitor events in logs
docker compose logs -f backend | grep "hfos.mode_changed\|governor.level_changed"
```

### Success Criteria
- All 4 events defined in EventBus
- Publishers added to correct modules
- Events logged in production monitoring

---

## TASK P1-2: Implement PolicyStore Cache Invalidation (45 minutes)

### Issue
PolicyStore reads have 2-3s lag during normal operations due to JSON file deserialization. Critical path (model promotion) uses atomic lock, but non-critical reads are slow.

### Impact
- Dashboard shows stale policy data (2-3s lag)
- Slow response time for policy queries
- 10-15% of CPU time spent on repeated JSON parsing

### Implementation

#### File: `backend/services/policy_store_v2.py`

Add in-memory cache with Redis-based invalidation:

```python
from datetime import datetime, timedelta
import asyncio

class PolicyStoreV2:
    def __init__(self):
        self.redis_client = None
        self._cache = {}           # ADD: In-memory cache
        self._cache_ttl = 10       # ADD: 10s TTL
        self._cache_timestamps = {} # ADD: Track cache age
        
    async def initialize(self):
        self.redis_client = await aioredis.from_url("redis://quantum_redis:6379")
        
        # ADD: Subscribe to invalidation events
        await self._subscribe_invalidation()
        
    async def _subscribe_invalidation(self):
        """Subscribe to policy update events and invalidate cache."""
        pubsub = self.redis_client.pubsub()
        await pubsub.subscribe("policy:invalidate")
        
        async def listener():
            async for message in pubsub.listen():
                if message["type"] == "message":
                    policy_name = message["data"].decode()
                    self._invalidate_cache(policy_name)
                    logger.debug(f"Cache invalidated for {policy_name}")
        
        asyncio.create_task(listener())
    
    def _invalidate_cache(self, policy_name: str = None):
        """Invalidate specific policy or all cache."""
        if policy_name:
            self._cache.pop(policy_name, None)
            self._cache_timestamps.pop(policy_name, None)
        else:
            self._cache.clear()
            self._cache_timestamps.clear()
    
    async def get_policy(self, name: str) -> dict:
        """Get policy with caching."""
        # Check cache first
        if name in self._cache:
            age = (datetime.utcnow() - self._cache_timestamps[name]).total_seconds()
            if age < self._cache_ttl:
                logger.debug(f"Cache hit for {name} (age: {age:.1f}s)")
                return self._cache[name]
        
        # Cache miss - read from Redis/disk
        policy = await self._read_policy_from_storage(name)
        
        # Update cache
        self._cache[name] = policy
        self._cache_timestamps[name] = datetime.utcnow()
        
        return policy
    
    async def update_policy(self, name: str, policy: dict):
        """Update policy and broadcast invalidation."""
        await self._write_policy_to_storage(name, policy)
        
        # Broadcast cache invalidation to all backend instances
        await self.redis_client.publish("policy:invalidate", name)
        
        # Invalidate local cache
        self._invalidate_cache(name)
```

### Testing

```bash
# Test cache performance
docker compose exec backend python -c "
from backend.services.policy_store_v2 import get_policy_store
import asyncio
import time

async def test():
    store = get_policy_store()
    
    # Cold read (cache miss)
    start = time.time()
    await store.get_policy('orchestrator')
    cold_time = time.time() - start
    print(f'Cold read: {cold_time*1000:.1f}ms')
    
    # Warm read (cache hit)
    start = time.time()
    await store.get_policy('orchestrator')
    warm_time = time.time() - start
    print(f'Warm read: {warm_time*1000:.1f}ms')
    
    # Should be <1ms for cached reads
    assert warm_time < 0.001, 'Cache not working'
    print('✓ Cache validated')

asyncio.run(test())
"
```

### Success Criteria
- Cache hit rate >90% after 5 minutes of operation
- Cached reads <1ms (vs 2-3s uncached)
- Invalidation works across all backend instances

---

## TASK P1-3: Complete Federation v2 Deprecation (90 minutes)

### Issue
Federation v2 bridge is operational, but legacy v2 code still exists in codebase. This creates confusion and technical debt.

### Impact
- ~2500 lines of unused code
- Potential for accidental v2 usage
- Harder to understand system architecture

### Implementation

#### Step 1: Verify v2 Bridge Operational (10 min)

```bash
# Ensure bridge is handling all v2 traffic
docker compose logs backend | grep "federation_v2_event_bridge"

# Should see events being bridged
grep "broadcast_to_v2_nodes" logs/backend.log
```

#### Step 2: Remove Legacy Files (30 min)

Delete or archive these files:

```
backend/federation/federated_engine_v2.py         # 450 lines - KEEP (still used by bridge)
backend/federation/v2_protocol.py                 # 280 lines - DELETE
backend/federation/v2_consensus.py                # 390 lines - DELETE
backend/federation/v2_node_discovery.py           # 220 lines - DELETE
backend/federation/v2_heartbeat.py                # 180 lines - DELETE
backend/tests/federation/test_v2_*.py             # 8 files - ARCHIVE
```

**Important:** Keep `federated_engine_v2.py` as it's used by the event bridge.

#### Step 3: Update Imports (20 min)

Scan for v2 imports and remove:

```bash
# Find all v2 imports
grep -r "from.*federation.*v2" backend/ --include="*.py"

# Remove imports from non-bridge files
# (Manual review required)
```

#### Step 4: Update Documentation (15 min)

Update architecture docs to reflect v3-only system:

- `AI_TRADING_ARCHITECTURE.md`
- `ARCHITECTURE_V2_INTEGRATION_COMPLETE.md`
- `AI_OS_INTEGRATION_SUMMARY.md`

Add deprecation notices:

```markdown
## Federation Architecture

**Current:** Federation v3 (EventBus-based)
**Deprecated:** Federation v2 (retained for bridge compatibility only)
**Bridge:** `federation_v2_event_bridge.py` maintains v2 protocol compatibility

See `backend/federation/federation_v2_event_bridge.py` for v2→v3 translation.
```

#### Step 5: Add Deprecation Warnings (15 min)

Add runtime warnings to remaining v2 code:

```python
# In federated_engine_v2.py
import warnings

class FederatedEngineV2:
    def __init__(self):
        warnings.warn(
            "FederatedEngineV2 is deprecated and maintained only for bridge compatibility. "
            "Use FederationV3 (EventBus-based) for new code.",
            DeprecationWarning,
            stacklevel=2
        )
```

### Testing

```bash
# Verify no v2 direct usage (bridge only)
docker compose logs backend | grep "FederatedEngineV2" | grep -v "bridge"
# Should return no results

# Check bridge still functional
docker compose logs backend | grep "federation_v2_event_bridge.*broadcast"
# Should see event bridging activity

# Run integration tests
python -m pytest backend/tests/federation/test_federation_v3.py -v
```

### Success Criteria
- All v2-specific files removed except bridge dependencies
- No direct v2 usage (only via bridge)
- Documentation updated with v3 as primary
- Deprecation warnings in remaining v2 code

---

## DEPLOYMENT CHECKLIST

### Pre-Maintenance
- [ ] Schedule maintenance window (recommend low-traffic hours)
- [ ] Notify stakeholders of brief downtime (<5 min)
- [ ] Create full system backup
- [ ] Verify rollback procedure tested

### During Maintenance
- [ ] Stop backend: `docker compose stop backend`
- [ ] Implement P1-1: Add events (30 min)
- [ ] Implement P1-2: Cache invalidation (45 min)
- [ ] Implement P1-3: Federation cleanup (90 min)
- [ ] Run test suite: `pytest backend/tests/ -v`
- [ ] Restart backend: `docker compose start backend`
- [ ] Verify health: `curl http://localhost:8000/health`

### Post-Maintenance Validation
- [ ] Monitor logs for new events (P1-1)
- [ ] Verify cache hit rate >90% (P1-2)
- [ ] Confirm no v2 direct usage (P1-3)
- [ ] Check system performance metrics
- [ ] Run 24h stability check

---

## ROLLBACK PLAN

If issues arise during implementation:

1. **Stop backend:** `docker compose stop backend`
2. **Restore pre-maintenance backup:**
   ```bash
   cp backups/pre-p1-maintenance/.env .env
   git checkout backend/
   ```
3. **Restart:** `docker compose start backend`
4. **Verify health:** `curl http://localhost:8000/health`

---

## ESTIMATED TIMELINE

| Task | Duration | Cumulative |
|------|----------|------------|
| P1-1: Missing Events | 30 min | 30 min |
| P1-2: Cache Invalidation | 45 min | 1h 15min |
| P1-3: Federation v2 Cleanup | 90 min | 2h 45min |
| Testing & Validation | 15 min | 3h 00min |

**Total:** 3 hours (with buffer)

---

## CONTACT

For questions or issues during implementation:
- Review IB-C-3 consistency sweep results
- Check `PRODUCTION_MONITORING.md` for debugging
- Escalate P0 issues immediately (see monitoring doc)

---

**Status:** Ready for first maintenance window  
**Priority:** P1 (Non-blocking)  
**Approval:** Recommended for next scheduled maintenance

