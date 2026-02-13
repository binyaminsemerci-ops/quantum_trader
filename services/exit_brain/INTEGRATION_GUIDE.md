# Exit Brain Heartbeat Integration Guide

## Hvordan legge til heartbeat i eksisterende Exit Brain

### Metode 1: Mixin (Anbefalt)

```python
# I din exit_brain.py, legg til import:
from services.exit_brain.heartbeat_mixin import HeartbeatMixin

# Endre klasse-definisjon:
class ExitBrainService(HeartbeatMixin):  # Legg til HeartbeatMixin
    def __init__(self, redis_client, config=None):
        HeartbeatMixin.__init__(self)  # Initialiser mixin
        
        # Din eksisterende init-kode...
        self.redis = redis_client
        self.active_positions = {}
        self.last_decision_ts = 0.0
        
        # Initialiser heartbeat
        self.init_heartbeat(redis_client, interval=1.0)
    
    async def run(self):
        """Hovedløkke"""
        self.start_heartbeat()  # Start heartbeat
        
        try:
            while self.running:
                start = time.time()
                
                # Din eksisterende beslutningslogikk...
                await self._process_positions()
                
                # Oppdater cycle time for heartbeat
                cycle_ms = int((time.time() - start) * 1000)
                self.set_cycle_time(cycle_ms)
                
                # Oppdater decision timestamp
                self.last_decision_ts = time.time()
                
                await asyncio.sleep(1)
        finally:
            await self.stop_heartbeat()  # Stopp heartbeat
```

### Metode 2: Direkte integrasjon

```python
import time
import asyncio

HEARTBEAT_STREAM = "quantum:stream:exit_brain.heartbeat"

class ExitBrainService:
    def __init__(self):
        self.redis = None
        self.active_positions = {}
        self.last_decision_ts = 0.0
        self._last_cycle_ms = 0
        self._heartbeat_task = None
    
    async def start(self):
        # Start heartbeat som bakgrunnsoppgave
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        # Din eksisterende start-kode...
    
    async def stop(self):
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        
        # Din eksisterende stop-kode...
    
    async def _heartbeat_loop(self):
        """Publiser heartbeat hvert sekund"""
        while True:
            try:
                await self._publish_heartbeat()
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
            
            await asyncio.sleep(1.0)
    
    async def _publish_heartbeat(self):
        """Publiser en heartbeat"""
        status = self._determine_status()
        
        await self.redis.xadd(
            HEARTBEAT_STREAM,
            {
                "timestamp": str(time.time()),
                "status": status,
                "active_positions_count": str(len(self.active_positions)),
                "last_decision_ts": str(self.last_decision_ts),
                "loop_cycle_ms": str(self._last_cycle_ms),
                "pending_exits": str(len(getattr(self, 'pending_exits', [])))
            },
            maxlen=1000
        )
    
    def _determine_status(self) -> str:
        """Bestem om vi er OK eller DEGRADED"""
        # Sjekk beslutningsloop
        if self._last_cycle_ms > 500:
            return "DEGRADED"
        
        # Sjekk om beslutninger stagnerer
        if len(self.active_positions) > 0:
            if time.time() - self.last_decision_ts > 10:
                return "DEGRADED"
        
        return "OK"
    
    async def process_cycle(self):
        """Hovedbeslutningssyklus - kall denne i din loop"""
        start = time.time()
        
        # Din eksisterende prosesseringslogikk...
        
        # Oppdater metrics for heartbeat
        self._last_cycle_ms = int((time.time() - start) * 1000)
        self.last_decision_ts = time.time()
```

### Viktige Attributter

Exit Brain MÅ ha disse attributtene for heartbeat å fungere:

| Attributt | Type | Beskrivelse |
|-----------|------|-------------|
| `active_positions` | dict/list | Aktive posisjoner |
| `last_decision_ts` | float | Timestamp for siste beslutning |
| `redis` | Redis client | For å publisere heartbeat |

### Testing

```python
# Sjekk at heartbeat publiseres
import redis
r = redis.Redis()

# Les siste heartbeat
messages = r.xrevrange("quantum:stream:exit_brain.heartbeat", count=1)
print(messages)

# Forventet output:
# [('1707840000123-0', {
#     b'timestamp': b'1707840000.123',
#     b'status': b'OK',
#     b'active_positions_count': b'3',
#     b'last_decision_ts': b'1707839999.456'
# })]
```

### Integrasjon med ExitBrain v3.5

For `microservices/exitbrain_v3_5/exit_brain.py`:

```python
# Legg til i __init__:
self._heartbeat_interval = 1.0
self._heartbeat_task = None
self._last_cycle_ms = 0
self.last_decision_ts = 0.0

# Legg til metode:
async def start_heartbeat_loop(self):
    """Start heartbeat background task"""
    async def heartbeat_loop():
        while True:
            try:
                if self.redis:
                    await self.redis.xadd(
                        "quantum:stream:exit_brain.heartbeat",
                        {
                            "timestamp": str(time.time()),
                            "status": "OK",  # Add degraded detection
                            "active_positions_count": str(self.plans_generated),
                            "last_decision_ts": str(self.last_decision_ts),
                            "loop_cycle_ms": str(self._last_cycle_ms)
                        },
                        maxlen=1000
                    )
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
            await asyncio.sleep(self._heartbeat_interval)
    
    self._heartbeat_task = asyncio.create_task(heartbeat_loop())
```

## Verifisering

1. Start Exit Brain
2. Sjekk heartbeat stream:
   ```bash
   redis-cli XLEN quantum:stream:exit_brain.heartbeat
   # Bør øke hvert sekund
   ```

3. Start Watchdog:
   ```bash
   systemctl start quantum-exit-brain-watchdog
   ```

4. Sjekk Watchdog logger:
   ```bash
   journalctl -u quantum-exit-brain-watchdog -f
   # Bør vise "Status: heartbeat=X.Xs ago, status=OK"
   ```
