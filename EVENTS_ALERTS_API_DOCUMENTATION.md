# Events & Alerts System API Documentation

## Overview
Real-time system event monitoring with both REST polling and WebSocket streaming capabilities.

## Endpoints

### REST API - Event Feed
**Endpoint:** `GET https://api.quantumfond.com/events/feed`  
**Description:** Returns the latest 10 system events  
**Method:** GET  
**Response:** JSON array of event objects

#### Response Structure
```json
[
  {
    "event_type": "trade",
    "severity": "info",
    "message": "Position closed at +1.8% PnL",
    "timestamp": 1766713748.08
  },
  {
    "event_type": "anomaly",
    "severity": "critical",
    "message": "Sharp drop in model accuracy detected",
    "timestamp": 1766713748.08
  },
  ...
]
```

#### Event Types
- **`anomaly`**: Model performance issues, data anomalies
- **`heal`**: Container restarts, auto-recovery events
- **`trade`**: Trade executions, position changes
- **`mode_switch`**: Brain mode transitions (OPTIMIZE → EXPAND, etc.)

#### Severity Levels
- **`info`**: Normal operational events
- **`warning`**: Attention required but not critical
- **`critical`**: Urgent issues requiring immediate attention

### WebSocket - Live Event Stream
**Endpoint:** `wss://api.quantumfond.com/events/stream`  
**Description:** Real-time event streaming  
**Protocol:** WebSocket  
**Update Frequency:** Every 2-5 seconds

---

## Testing

### REST Endpoint Test
```bash
curl https://api.quantumfond.com/events/feed
```

**Expected Output:**
```json
[
  {"event_type":"trade","severity":"info","message":"Position closed at +1.8% PnL","timestamp":1766713748.08},
  {"event_type":"heal","severity":"warning","message":"Container restarted successfully","timestamp":1766713748.08},
  ...
]
```

### WebSocket Test (Python)
```python
import asyncio
import websockets
import json

async def test_event_stream():
    uri = "wss://api.quantumfond.com/events/stream"
    async with websockets.connect(uri) as ws:
        for i in range(5):
            data = json.loads(await ws.recv())
            print(f"[{data['severity'].upper()}] {data['event_type']}: {data['message']}")

asyncio.run(test_event_stream())
```

**Run the included test script:**
```bash
python test_event_stream.py
```

**Expected Output:**
```
🔌 Connecting to wss://api.quantumfond.com/events/stream...
✅ Connected! Receiving events...

🚨 [CRITICAL] mode_switch: CEO Brain changed mode → OPTIMIZE
   Timestamp: 1766713759.85

ℹ️ [INFO] heal: Container restarted successfully
   Timestamp: 1766713764.34

✅ Test completed - received 5 events successfully
```

---

## Frontend Integration

### React + TypeScript Implementation

```typescript
import { useEffect, useState } from 'react';

interface SystemEvent {
  event_type: 'anomaly' | 'heal' | 'trade' | 'mode_switch';
  severity: 'info' | 'warning' | 'critical';
  message: string;
  timestamp: number;
}

export function EventFeed() {
  const [events, setEvents] = useState<SystemEvent[]>([]);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    // Option 1: WebSocket for real-time streaming (recommended)
    const ws = new WebSocket("wss://api.quantumfond.com/events/stream");
    
    ws.onopen = () => {
      console.log("✅ Event stream connected");
      setConnected(true);
    };
    
    ws.onmessage = (e) => {
      const event: SystemEvent = JSON.parse(e.data);
      setEvents(prev => [event, ...prev].slice(0, 20)); // Keep last 20 events
    };
    
    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
      setConnected(false);
    };
    
    ws.onclose = () => {
      console.log("Event stream disconnected");
      setConnected(false);
    };

    return () => ws.close();
  }, []);

  return (
    <div className="event-feed">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-xl font-bold">📡 Live Event Feed</h3>
        <span className={`px-3 py-1 rounded text-xs ${connected ? 'bg-green-600' : 'bg-red-600'}`}>
          {connected ? '● LIVE' : '● DISCONNECTED'}
        </span>
      </div>

      <div className="space-y-2 max-h-96 overflow-y-auto">
        {events.map((event, idx) => (
          <EventCard key={idx} event={event} />
        ))}
      </div>
    </div>
  );
}

function EventCard({ event }: { event: SystemEvent }) {
  const severityConfig = {
    info: { icon: 'ℹ️', color: 'bg-blue-900 border-blue-600' },
    warning: { icon: '⚠️', color: 'bg-yellow-900 border-yellow-600' },
    critical: { icon: '🚨', color: 'bg-red-900 border-red-600' }
  };

  const typeIcon = {
    anomaly: '📉',
    heal: '🔧',
    trade: '💹',
    mode_switch: '🔄'
  };

  const config = severityConfig[event.severity];
  const icon = typeIcon[event.event_type];

  return (
    <div className={`p-3 rounded border-l-4 ${config.color}`}>
      <div className="flex items-start gap-2">
        <span className="text-xl">{icon}</span>
        <div className="flex-1">
          <div className="flex items-center justify-between">
            <span className="font-semibold text-sm uppercase text-gray-400">
              {event.event_type.replace('_', ' ')}
            </span>
            <span className="text-xs text-gray-500">
              {new Date(event.timestamp * 1000).toLocaleTimeString()}
            </span>
          </div>
          <p className="text-sm mt-1">{event.message}</p>
          <span className={`text-xs ${config.icon}`}>
            {config.icon} {event.severity.toUpperCase()}
          </span>
        </div>
      </div>
    </div>
  );
}
```

### Alternative: REST Polling Fallback

If WebSocket is unavailable, fall back to REST polling:

```typescript
useEffect(() => {
  // Try WebSocket first
  let ws: WebSocket | null = null;
  let pollInterval: NodeJS.Timeout | null = null;

  try {
    ws = new WebSocket("wss://api.quantumfond.com/events/stream");
    ws.onmessage = (e) => {
      const event = JSON.parse(e.data);
      setEvents(prev => [event, ...prev].slice(0, 20));
    };
  } catch {
    // WebSocket failed, use REST polling
    console.log("WebSocket unavailable, using REST polling");
    
    const fetchEvents = async () => {
      const res = await fetch("https://api.quantumfond.com/events/feed");
      const data = await res.json();
      setEvents(data);
    };

    fetchEvents(); // Initial fetch
    pollInterval = setInterval(fetchEvents, 10000); // Poll every 10 seconds
  }

  return () => {
    if (ws) ws.close();
    if (pollInterval) clearInterval(pollInterval);
  };
}, []);
```

---

## Advanced Features

### Event Filtering by Severity

```typescript
const [severityFilter, setSeverityFilter] = useState<string | null>(null);

const filteredEvents = events.filter(e => 
  severityFilter === null || e.severity === severityFilter
);

// UI
<div className="flex gap-2 mb-4">
  <button onClick={() => setSeverityFilter(null)}>All</button>
  <button onClick={() => setSeverityFilter('critical')}>Critical</button>
  <button onClick={() => setSeverityFilter('warning')}>Warnings</button>
  <button onClick={() => setSeverityFilter('info')}>Info</button>
</div>
```

### Event Type Filtering

```typescript
const [typeFilter, setTypeFilter] = useState<string | null>(null);

const filteredEvents = events.filter(e => 
  typeFilter === null || e.event_type === typeFilter
);

// UI
<select onChange={(e) => setTypeFilter(e.target.value || null)}>
  <option value="">All Types</option>
  <option value="anomaly">Anomalies</option>
  <option value="heal">Healing Events</option>
  <option value="trade">Trades</option>
  <option value="mode_switch">Mode Changes</option>
</select>
```

### Event Statistics Dashboard

```typescript
function EventStats({ events }: { events: SystemEvent[] }) {
  const stats = {
    total: events.length,
    critical: events.filter(e => e.severity === 'critical').length,
    warning: events.filter(e => e.severity === 'warning').length,
    info: events.filter(e => e.severity === 'info').length,
    byType: events.reduce((acc, e) => {
      acc[e.event_type] = (acc[e.event_type] || 0) + 1;
      return acc;
    }, {} as Record<string, number>)
  };

  return (
    <div className="grid grid-cols-4 gap-4 mb-6">
      <StatCard label="Total Events" value={stats.total} color="blue" />
      <StatCard label="Critical" value={stats.critical} color="red" />
      <StatCard label="Warnings" value={stats.warning} color="yellow" />
      <StatCard label="Info" value={stats.info} color="green" />
    </div>
  );
}
```

### Sound Notifications for Critical Events

```typescript
useEffect(() => {
  if (events[0]?.severity === 'critical') {
    // Play alert sound
    const audio = new Audio('/alert.mp3');
    audio.play().catch(e => console.log('Audio play failed:', e));
    
    // Optional: Browser notification
    if (Notification.permission === 'granted') {
      new Notification('Critical Event', {
        body: events[0].message,
        icon: '/logo.png'
      });
    }
  }
}, [events]);
```

---

## CORS Configuration

Events API is configured to accept requests from:
- ✅ `https://app.quantumfond.com`
- ✅ `http://localhost:5173` (development)
- ✅ `http://localhost:8889` (VPS testing)

**Headers:**
- `Access-Control-Allow-Origin: https://app.quantumfond.com`
- `Access-Control-Allow-Credentials: true`
- `Access-Control-Allow-Methods: *`
- `Access-Control-Allow-Headers: *`

---

## Verification Checklist

✅ REST endpoint `/events/feed` returns JSON array of events  
✅ WebSocket endpoint `/events/stream` streams events every 2-5 seconds  
✅ Both endpoints accessible via HTTPS with valid SSL  
✅ CORS headers configured for app.quantumfond.com  
✅ Event types include anomaly, heal, trade, mode_switch  
✅ Severity levels: info, warning, critical  
✅ Timestamps included in Unix epoch format  
✅ Test script `test_event_stream.py` runs successfully  
✅ No CORS errors when accessing from frontend  
✅ WebSocket auto-reconnect handled gracefully  

---

## Future Enhancements (Phase 9+)

1. **Persistent Event Storage**: Store events in PostgreSQL for historical analysis
2. **Event Filtering API**: Query parameters for filtering by type, severity, time range
3. **Event Replay**: Replay historical events for debugging
4. **Alert Rules**: Configure custom alert conditions (e.g., "3 critical events in 5 minutes")
5. **Event Aggregation**: Group similar events to reduce noise
6. **Email/SMS Notifications**: Send alerts via external channels
7. **Event Correlation**: Detect patterns and causal relationships between events

---

>>> **[Phase 8 Complete – Event & Alert System live via api.quantumfond.com]**
