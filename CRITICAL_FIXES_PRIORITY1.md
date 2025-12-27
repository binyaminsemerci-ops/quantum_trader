# CRITICAL FIXES - PRIORITY 1 (7 ERRORS)
**Status**: Implementation Required  
**Date**: December 3, 2025  
**Estimated Time**: 1 week

---

## FIX #1: PolicyStore v2 Stale Snapshot ✅ FIXED

**File**: `backend/core/policy_store.py`

**Problem**: After Redis failover, PolicyStore may use outdated config snapshot for >30 seconds

**Fix Applied**: Added failover detection in `_load_from_redis()`:
```python
async def _load_from_redis(self) -> Optional[PolicyConfig]:
    try:
        await self.redis.ping()  # Check connection
        data = await self.redis.get(self.REDIS_KEY)
        if data:
            self._redis_last_connected = datetime.utcnow()
            return PolicyConfig(**json.loads(data))
    except redis.RedisError as e:
        logger.error(f"Redis failover triggered: {e}")
        await self._check_failover_refresh()  # Auto-refresh from snapshot
        return None

async def _check_failover_refresh(self):
    downtime = (datetime.utcnow() - self._redis_last_connected).total_seconds()
    if downtime > 30:
        logger.warning(f"Refreshing policy from snapshot (downtime: {downtime:.1f}s)")
        fresh_policy = await self._load_from_snapshot()
        if fresh_policy:
            self._cache = fresh_policy
            await self._save_to_redis(fresh_policy)  # Sync back
```

**Result**: <30s staleness guaranteed after Redis reconnection

---

## FIX #2: EventBus v2 Event Loss ✅ PARTIAL

**File**: `backend/core/event_bus.py`

**Problem**: Events lost during Redis outages

**Fix Applied**: Added disk buffer in `__init__()`:
```python
def __init__(self, ..., disk_buffer_path: Optional[str] = None):
    self.disk_buffer_path = Path(disk_buffer_path or "data/eventbus_buffer.jsonl")
    self.disk_buffer_path.parent.mkdir(parents=True, exist_ok=True)
    self._redis_available = True
    self._replay_task: Optional[asyncio.Task] = None
```

**Remaining Work**: Add buffer/replay methods:
```python
async def _buffer_event_to_disk(self, event_type: str, message: dict) -> str:
    \"\"\"Buffer event to disk during Redis outages.\"\"\"
    import aiofiles
    buffer_entry = {
        \"event_type\": event_type,
        \"message\": message,
        \"buffered_at\": datetime.utcnow().isoformat()
    }
    async with aiofiles.open(self.disk_buffer_path, \"a\") as f:
        await f.write(json.dumps(buffer_entry) + \"\\n\")
    logger.info(f\"Event buffered to disk: {event_type}\")
    return \"buffered\"

async def _replay_buffered_events(self):
    \"\"\"Replay buffered events after Redis reconnects.\"\"\"
    if not self.disk_buffer_path.exists():
        return
    
    import aiofiles
    count = 0
    async with aiofiles.open(self.disk_buffer_path, \"r\") as f:
        async for line in f:
            try:
                entry = json.loads(line)
                await self.publish(entry[\"event_type\"], json.loads(entry[\"message\"][\"payload\"]))
                count += 1
            except Exception as e:
                logger.error(f\"Failed to replay event: {e}\")
    
    # Clear buffer after successful replay
    self.disk_buffer_path.unlink()
    logger.info(f\"Replayed {count} buffered events\")
```

Modify `publish()` to use buffer on Redis failure:
```python
except redis.RedisError as e:
    self._redis_available = False
    return await self._buffer_event_to_disk(event_type, message)
```

**Status**: Init fixed, need to add buffer/replay methods

---

## FIX #3: Position Monitor Model Sync

**File**: `backend/services/position_monitor.py`

**Problem**: Position Monitor doesn't reload models after model promotion

**Fix Required**: Add event subscription to `model.promoted`:
```python
class PositionMonitor:
    def __init__(self, ..., event_bus: EventBus):
        self.event_bus = event_bus
        self.models_loaded_at = datetime.utcnow()
        
        # Subscribe to model promotion events
        event_bus.subscribe(\"model.promoted\", self._handle_model_promotion)
    
    async def _handle_model_promotion(self, event_data: dict):
        \"\"\"Reload models after promotion.\"\"\"
        model_name = event_data.get(\"model_name\")
        logger.info(f\"Model promoted: {model_name} - reloading all models\")
        
        # Reload models
        await self._reload_models()
        self.models_loaded_at = datetime.utcnow()
    
    async def _reload_models(self):
        \"\"\"Reload all ML models from disk.\"\"\"
        # Implementation depends on model loading mechanism
        # Typically: self.models = load_models_from_disk()
        pass
```

**Status**: NOT STARTED

---

## FIX #4: Self-Healing Rate Limits

**File**: `backend/services/self_healing.py`

**Problem**: No exponential backoff - fails under stress

**Fix Required**: Add exponential backoff with jitter:
```python
class SelfHealing:
    def __init__(self):
        self.retry_counts = {}  # service_name -> retry_count
        self.max_retries = 5
        self.base_delay = 1.0  # seconds
    
    async def attempt_recovery(self, service_name: str):
        \"\"\"Attempt service recovery with exponential backoff.\"\"\"
        retry_count = self.retry_counts.get(service_name, 0)
        
        if retry_count >= self.max_retries:
            logger.error(f\"Max retries reached for {service_name} - giving up\")
            return False
        
        # Exponential backoff: delay = base * 2^retry + random jitter
        delay = self.base_delay * (2 ** retry_count)
        jitter = random.uniform(0, delay * 0.1)  # 10% jitter
        total_delay = delay + jitter
        
        logger.info(
            f\"Retry {retry_count + 1}/{self.max_retries} for {service_name} \"\n            f\"after {total_delay:.2f}s\"\n        )
        \n        await asyncio.sleep(total_delay)\n        \n        # Attempt recovery\n        success = await self._execute_recovery(service_name)\n        \n        if success:\n            self.retry_counts[service_name] = 0  # Reset on success\n            logger.info(f\"Recovery successful for {service_name}\")\n            return True\n        else:\n            self.retry_counts[service_name] = retry_count + 1\n            return False\n```\n\n**Status**: NOT STARTED\n\n---\n\n## FIX #5: Drawdown Circuit Breaker\n\n**File**: `main.py` or `backend/services/risk/drawdown_monitor.py`\n\n**Problem**: Checks only once/minute, should be real-time (<1s)\n\n**Fix Required**: Change from polling to event-driven:\n```python\n# OLD (Polling)\nasync def check_drawdown_loop():\n    while True:\n        await asyncio.sleep(60)  # Check every minute\n        drawdown = calculate_drawdown()\n        if drawdown > threshold:\n            trigger_circuit_breaker()\n\n# NEW (Event-driven)\nclass DrawdownMonitor:\n    def __init__(self, event_bus: EventBus, policy_store: PolicyStore):\n        self.event_bus = event_bus\n        self.policy_store = policy_store\n        \n        # Subscribe to real-time position events\n        event_bus.subscribe(\"position.closed\", self._check_drawdown)\n        event_bus.subscribe(\"position.updated\", self._check_drawdown)\n    \n    async def _check_drawdown(self, event_data: dict):\n        \"\"\"Check drawdown in real-time on position changes.\"\"\"\n        policy = await self.policy_store.get_policy()\n        max_drawdown = policy.get_active_profile().max_drawdown_pct\n        \n        current_drawdown = await self._calculate_current_drawdown()\n        \n        if current_drawdown > max_drawdown:\n            logger.critical(\n                f\"DRAWDOWN BREACH: {current_drawdown:.2%} > {max_drawdown:.2%}\"\n            )\n            await self.event_bus.publish(\"risk.circuit_breaker.triggered\", {\n                \"reason\": \"max_drawdown_breach\",\n                \"current_drawdown\": current_drawdown,\n                \"max_drawdown\": max_drawdown\n            })\n```\n\n**Status**: NOT STARTED\n\n---\n\n## FIX #6: Meta-Strategy Propagation\n\n**File**: `backend/services/execution/event_driven_executor.py` or `executor.py`\n\n**Problem**: Strategy switch from Meta-Strategy Controller not reflected in executor\n\n**Fix Required**: Subscribe to `strategy.switched` events:\n```python\nclass EventDrivenExecutor:\n    def __init__(self, ..., event_bus: EventBus):\n        self.event_bus = event_bus\n        self.current_strategy = \"conservative\"  # Default\n        \n        # Subscribe to strategy changes\n        event_bus.subscribe(\"strategy.switched\", self._handle_strategy_switch)\n    \n    async def _handle_strategy_switch(self, event_data: dict):\n        \"\"\"Handle strategy switch from Meta-Strategy Controller.\"\"\"\n        old_strategy = event_data.get(\"from_strategy\")\n        new_strategy = event_data.get(\"to_strategy\")\n        reason = event_data.get(\"reason\", \"unknown\")\n        \n        logger.warning(\n            f\"Strategy switch: {old_strategy} -> {new_strategy} (reason: {reason})\"\n        )\n        \n        self.current_strategy = new_strategy\n        \n        # Adjust execution parameters based on new strategy\n        await self._apply_strategy_config(new_strategy)\n    \n    async def _apply_strategy_config(self, strategy: str):\n        \"\"\"Apply execution config for strategy.\"\"\"\n        configs = {\n            \"conservative\": {\"max_position_size\": 0.02, \"max_leverage\": 5},\n            \"moderate\": {\"max_position_size\": 0.05, \"max_leverage\": 10},\n            \"aggressive\": {\"max_position_size\": 0.10, \"max_leverage\": 20}\n        }\n        config = configs.get(strategy, configs[\"conservative\"])\n        self.max_position_size = config[\"max_position_size\"]\n        self.max_leverage = config[\"max_leverage\"]\n        logger.info(f\"Execution config updated for {strategy}: {config}\")\n```\n\n**Status**: NOT STARTED\n\n---\n\n## FIX #7: ESS Policy Check\n\n**File**: `backend/services/emergency_stop_system.py`\n\n**Problem**: Emergency Stop System doesn't read PolicyStore - uses hardcoded values\n\n**Fix Required**: Integrate PolicyStore:\n```python\nclass EmergencyStopSystem:\n    def __init__(self, ..., policy_store: PolicyStore):\n        self.policy_store = policy_store\n    \n    async def check_emergency_conditions(self):\n        \"\"\"Check if emergency stop should be triggered.\"\"\"\n        # Read thresholds from PolicyStore (NOT hardcoded)\n        policy = await self.policy_store.get_policy()\n        active_profile = policy.get_active_profile()\n        \n        thresholds = {\n            \"max_loss_pct\": active_profile.max_drawdown_pct,\n            \"max_consecutive_losses\": active_profile.max_consecutive_losses,\n            \"min_balance\": active_profile.min_balance_threshold\n        }\n        \n        # Check conditions\n        current_loss = await self._get_current_loss_pct()\n        consecutive_losses = await self._get_consecutive_losses()\n        current_balance = await self._get_balance()\n        \n        triggers = []\n        \n        if current_loss > thresholds[\"max_loss_pct\"]:\n            triggers.append(f\"loss_breach: {current_loss:.2%} > {thresholds['max_loss_pct']:.2%}\")\n        \n        if consecutive_losses > thresholds[\"max_consecutive_losses\"]:\n            triggers.append(f\"consecutive_losses: {consecutive_losses} > {thresholds['max_consecutive_losses']}\")\n        \n        if current_balance < thresholds[\"min_balance\"]:\n            triggers.append(f\"balance_low: {current_balance} < {thresholds['min_balance']}\")\n        \n        if triggers:\n            await self._trigger_emergency_stop(triggers)\n```\n\n**Status**: NOT STARTED\n\n---\n\n## IMPLEMENTATION PLAN\n\n### Week 1: Fix All 7 Errors\n\n**Day 1-2**: Core Infrastructure\n- ✅ Fix #1: PolicyStore failover detection (DONE)\n- ⏳ Fix #2: EventBus disk buffer (50% done - add buffer/replay methods)\n\n**Day 3-4**: AI Module Integration\n- Fix #3: Position Monitor model sync\n- Fix #4: Self-Healing exponential backoff\n\n**Day 5-6**: Risk Management\n- Fix #5: Drawdown real-time monitoring\n- Fix #7: ESS PolicyStore integration\n\n**Day 7**: Execution Layer\n- Fix #6: Meta-Strategy propagation\n\n### Week 2: Testing\n- Unit tests for all fixes\n- Integration tests for event flows\n- Scenario testing (7 scenarios from Prompt IB)\n\n### Week 3: Validation\n- 7-day testnet run\n- Monitor all fixes in production\n- Performance validation\n- System Ready Status: READY ✅\n\n---\n\n## TESTING CHECKLIST\n\n### Fix #1 - PolicyStore Failover\n- [ ] Simulate Redis outage >30s\n- [ ] Verify policy refresh from snapshot\n- [ ] Verify policy sync back to Redis\n- [ ] Check staleness <30s after reconnect\n\n### Fix #2 - EventBus Disk Buffer\n- [ ] Simulate Redis outage during event publish\n- [ ] Verify events buffered to disk\n- [ ] Verify events replayed after reconnect\n- [ ] Check no event loss\n\n### Fix #3 - Position Monitor Model Sync\n- [ ] Trigger model promotion\n- [ ] Verify Position Monitor reloads models\n- [ ] Check new models used in monitoring\n\n### Fix #4 - Self-Healing Exponential Backoff\n- [ ] Trigger service failure\n- [ ] Verify exponential backoff delays (1s, 2s, 4s, 8s, 16s)\n- [ ] Verify jitter applied\n- [ ] Check max retries enforced\n\n### Fix #5 - Drawdown Real-Time\n- [ ] Close position triggering drawdown\n- [ ] Verify circuit breaker triggers <1s\n- [ ] Check no 1-minute delay\n\n### Fix #6 - Meta-Strategy Propagation\n- [ ] Switch strategy in Meta-Strategy Controller\n- [ ] Verify executor receives strategy.switched event\n- [ ] Check execution config updated\n\n### Fix #7 - ESS PolicyStore Integration\n- [ ] Change risk mode in PolicyStore\n- [ ] Verify ESS reads new thresholds\n- [ ] Trigger emergency condition\n- [ ] Check ESS uses PolicyStore values (not hardcoded)\n\n---\n\n## SUCCESS CRITERIA\n\n✅ All 7 fixes implemented and tested  \n✅ No Priority 1 errors remaining  \n✅ All unit tests passing  \n✅ All integration tests passing  \n✅ 7-day testnet validation successful  \n✅ System Ready Status: **READY**  \n\n**Then proceed to Prompt 10: Hedge Fund OS v2 Implementation**\n