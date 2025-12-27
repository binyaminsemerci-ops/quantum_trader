"""
Check Exit Brain V3 state via HTTP endpoint.
"""
import requests
import json

# Call backend endpoint to get state
response = requests.get("http://localhost:8000/admin/exit-brain-state")

if response.status_code != 200:
    print(f"❌ Error: {response.status_code}")
    print(response.text)
else:
    data = response.json()
    
    print("\n" + "="*80)
    print("EXIT BRAIN V3 - STATE INSPECTION")
    print("="*80)
    
    if 'error' in data:
        print(f"\n❌ Error: {data['error']}")
    elif 'positions' not in data or not data['positions']:
        print("\n❌ No positions in state!")
    else:
        positions = data['positions']
        print(f"\n✅ Found {len(positions)} positions in state\n")
        
        for pos in positions:
            print(f"{'─'*80}")
            print(f"💰 {pos['symbol']} {pos['side']}")
            print(f"{'─'*80}")
            print(f"Entry:        ${pos['entry_price']:.4f}")
            print(f"Size:         {pos['size']}")
            print(f"Active SL:    ${pos['active_sl']:.4f}" if pos.get('active_sl') else "Active SL:    None")
            
            if pos.get('tp_legs'):
                print(f"\n✅ TP Legs: {len(pos['tp_legs'])}")
                for i, leg in enumerate(pos['tp_legs']):
                    triggered = "✅ TRIGGERED" if i in pos.get('triggered_legs', []) else "⏳ Pending"
                    print(f"  TP{i}: ${leg['price']:.4f} ({leg['size_pct']*100:.1f}%) - {triggered}")
            else:
                print(f"\n❌ NO TP LEGS!")
            
            print(f"\nTriggered legs: {pos.get('triggered_legs', [])}")
            print(f"Closed size: {pos.get('closed_size', 0)}")
            print()
    
    print("="*80)
