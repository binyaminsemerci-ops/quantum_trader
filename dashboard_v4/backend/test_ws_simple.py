"""
Simple WebSocket test using Python built-in websocket
"""
import json
import time
from websocket import create_connection

print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("🔌 WEBSOCKET STREAM TEST (Alternative Client)")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

try:
    ws = create_connection("ws://localhost:8000/stream/live")
    print("✅ Connected successfully!")
    print()
    print("📊 Receiving live updates (5 messages):")
    print("━" * 80)
    
    for i in range(5):
        result = ws.recv()
        data = json.loads(result)
        
        print(f"\n📦 Update #{i+1} - {time.strftime('%H:%M:%S', time.localtime(data['timestamp']))}")
        print(f"   🖥️  System:  CPU {data['cpu']}% | RAM {data['ram']}%")
        print(f"   🤖 AI:      Accuracy {data['accuracy']*100:.1f}% | Latency {data['latency']}ms")
        print(f"   💼 Portfolio: PnL ${data['pnl']:,.2f}")
    
    ws.close()
    
    print()
    print("━" * 80)
    print("✅ Test completed successfully!")
    print()
    print(">>> [Phase 4 Complete – Real-time stream operational and stable]")
    
except Exception as e:
    print(f"❌ Connection failed: {e}")
    print()
    print(">>> [Test failed - check backend logs]")
