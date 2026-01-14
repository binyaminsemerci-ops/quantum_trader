from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio
import random
import time

router = APIRouter(prefix="/events", tags=["Events & Alerts"])

EVENT_TYPES = ["anomaly", "heal", "trade", "mode_switch"]
SEVERITY = ["info", "warning", "critical"]

def generate_event():
    """Generate a simulated system event"""
    event_type = random.choice(EVENT_TYPES)
    severity = random.choice(SEVERITY)
    msg = {
        "anomaly": "Sharp drop in model accuracy detected",
        "heal": "Container restarted successfully",
        "trade": "Position closed at +1.8% PnL",
        "mode_switch": "CEO Brain changed mode → OPTIMIZE"
    }[event_type]
    return {
        "event_type": event_type,
        "severity": severity,
        "message": msg,
        "timestamp": time.time()
    }

# REST endpoint for latest events
@router.get("/feed")
async def get_latest_events():
    """
    Return 10 simulated system events
    
    Returns:
        List of recent system events with type, severity, message, and timestamp
    """
    return [generate_event() for _ in range(10)]

# WebSocket for live streaming of events
@router.websocket("/stream")
async def event_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time event streaming
    
    Sends system events every 2-5 seconds to connected clients
    """
    await websocket.accept()
    try:
        while True:
            event = generate_event()
            await websocket.send_json(event)
            await asyncio.sleep(random.uniform(2.0, 5.0))
    except WebSocketDisconnect:
        print("Event stream disconnected")
