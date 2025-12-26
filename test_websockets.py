"""
WebSocket Connection Test for Quantum Fund Dashboard
Tests wss://api.quantumfond.com/stream/live and /events/stream
"""
import asyncio
import websockets
import json
import sys

async def test_live_stream():
    """Test live data stream WebSocket"""
    uri = "wss://api.quantumfond.com/stream/live"
    print(f"🔌 Connecting to {uri}...")
    
    try:
        async with websockets.connect(uri, ssl=True) as websocket:
            print("✅ Connected to live stream!")
            
            # Receive 3 messages
            for i in range(3):
                message = await asyncio.wait_for(websocket.recv(), timeout=10)
                data = json.loads(message)
                print(f"📊 Message {i+1}: {json.dumps(data, indent=2)}")
            
            print("✅ Live stream test passed\n")
            return True
    except asyncio.TimeoutError:
        print("❌ Timeout waiting for live stream data\n")
        return False
    except Exception as e:
        print(f"❌ Live stream error: {e}\n")
        return False

async def test_events_stream():
    """Test events stream WebSocket"""
    uri = "wss://api.quantumfond.com/events/stream"
    print(f"🔌 Connecting to {uri}...")
    
    try:
        async with websockets.connect(uri, ssl=True) as websocket:
            print("✅ Connected to events stream!")
            
            # Receive 3 messages
            for i in range(3):
                message = await asyncio.wait_for(websocket.recv(), timeout=10)
                data = json.loads(message)
                print(f"🚨 Event {i+1}: {json.dumps(data, indent=2)}")
            
            print("✅ Events stream test passed\n")
            return True
    except asyncio.TimeoutError:
        print("❌ Timeout waiting for events stream data\n")
        return False
    except Exception as e:
        print(f"❌ Events stream error: {e}\n")
        return False

async def main():
    """Run all WebSocket tests"""
    print("=" * 60)
    print("WebSocket Connection Test Suite")
    print("=" * 60 + "\n")
    
    results = []
    
    # Test live stream
    print("TEST 1: Live Data Stream")
    print("-" * 60)
    results.append(await test_live_stream())
    
    # Test events stream
    print("TEST 2: Events Stream")
    print("-" * 60)
    results.append(await test_events_stream())
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if all(results):
        print("\n🎉 All WebSocket tests passed!")
        sys.exit(0)
    else:
        print("\n⚠️ Some WebSocket tests failed")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
