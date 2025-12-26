"""
Test script for WebSocket event stream
Usage: python test_event_stream.py
"""
import asyncio
import websockets
import json

async def test_event_stream():
    uri = "wss://api.quantumfond.com/events/stream"
    print(f"🔌 Connecting to {uri}...")
    
    try:
        async with websockets.connect(uri) as ws:
            print("✅ Connected! Receiving events...\n")
            
            for i in range(5):
                data = json.loads(await ws.recv())
                severity_icon = {
                    "info": "ℹ️",
                    "warning": "⚠️", 
                    "critical": "🚨"
                }.get(data['severity'], "•")
                
                print(f"{severity_icon} [{data['severity'].upper()}] {data['event_type']}: {data['message']}")
                print(f"   Timestamp: {data['timestamp']}\n")
                
            print("✅ Test completed - received 5 events successfully")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_event_stream())
