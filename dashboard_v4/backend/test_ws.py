"""
WebSocket Test Client
Tests the real-time streaming endpoint for Quantum Trader Dashboard
"""
import asyncio
import websockets
import json
from datetime import datetime


async def test_websocket():
    """Connect to WebSocket and receive live updates"""
    uri = "ws://localhost:8000/stream/live"
    
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("🔌 WEBSOCKET STREAM TEST")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"Connecting to: {uri}")
    print()
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected successfully!")
            print()
            print("📊 Receiving live updates (5 messages):")
            print("━" * 80)
            
            for i in range(5):
                message = await websocket.recv()
                data = json.loads(message)
                
                print(f"\n📦 Update #{i+1} - {datetime.fromtimestamp(data['timestamp']).strftime('%H:%M:%S')}")
                print(f"   🖥️  System:  CPU {data['cpu']}% | RAM {data['ram']}% | {data['services']} services")
                print(f"   🤖 AI:      Accuracy {data['accuracy']*100:.1f}% | Latency {data['latency']}ms | Sharpe {data['sharpe']}")
                print(f"   💼 Portfolio: PnL ${data['pnl']:,.2f} | {data['positions']} positions | {data['exposure']*100:.1f}% exposure")
            
            print()
            print("━" * 80)
            print("✅ Test completed successfully!")
            print("   Server handled 5 messages without errors")
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    print()
    success = asyncio.run(test_websocket())
    print()
    
    if success:
        print(">>> [Phase 4 Complete – Real-time stream operational and stable]")
    else:
        print(">>> [Test failed - check if backend is running on port 8000]")
    print()
