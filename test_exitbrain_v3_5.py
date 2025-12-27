#!/usr/bin/env python3
"""
ExitBrain v3.5 Validation Test
Tests ExitBrain calculations for test positions in Redis
"""

import os
import sys
import time
import redis
import json

# Add paths for imports
sys.path.insert(0, '/app/microservices')
sys.path.insert(0, '/app')

from exitbrain_v3_5.exit_brain import ExitBrainV35, SignalContext

def main():
    print("🚀 ExitBrain v3.5 Validation Test")
    print("=" * 60)
    
    # Connect to Redis
    r = redis.Redis(host=os.getenv("REDIS_HOST", "redis"), port=6379, db=0)
    print("✅ Connected to Redis")
    
    # Initialize ExitBrain
    brain = ExitBrainV35()
    print("✅ ExitBrain v3.5 initialized")
    print()
    
    # Run test loop
    for loop_num in range(2):
        print(f"\n[Loop {loop_num + 1}/2]")
        print("-" * 60)
        
        # Get test positions
        positions = [p.decode() for p in r.keys("quantum:positions:test:*")]
        print(f"Found {len(positions)} test positions")
        
        if not positions:
            print("⚠️ No test positions found!")
            break
        
        for pos_key in positions:
            # Get position data
            data = r.hgetall(pos_key)
            sym = pos_key.split(":")[-1]
            side = data[b'side'].decode()
            entry = float(data[b'entry'])
            mark = float(data[b'mark'])
            pnl = float(data[b'pnl'])
            leverage = float(data[b'leverage'])
            
            print(f"\n📊 Testing {sym} ({side.upper()}):")
            print(f"   Entry: ${entry:,.2f}")
            print(f"   Mark:  ${mark:,.2f}")
            print(f"   P&L:   {pnl:+.2f}%")
            print(f"   Leverage: {leverage:.1f}x")
            
            # Create signal context
            ctx = SignalContext(
                symbol=sym,
                side=side,
                confidence=0.75,
                entry_price=entry,
                atr_value=entry * 0.02,  # 2% of entry as ATR
                timestamp=time.time()
            )
            
            try:
                # Calculate exits
                exit_plan = brain.calculate_exits(ctx)
                
                print(f"   ✅ ExitBrain Decision:")
                print(f"      Leverage:     {exit_plan.leverage:.1f}x")
                print(f"      Take Profit:  {exit_plan.take_profit_pct:+.2f}%")
                print(f"      Stop Loss:    {exit_plan.stop_loss_pct:+.2f}%")
                print(f"      Trailing:     {'Yes' if exit_plan.use_trailing else 'No'}")
                
                # Store decision in Redis for verification
                decision_key = f"quantum:exit_decision:{sym}"
                r.setex(
                    decision_key,
                    300,  # Expire after 5 minutes
                    json.dumps({
                        "symbol": sym,
                        "leverage": exit_plan.leverage,
                        "tp_pct": exit_plan.take_profit_pct,
                        "sl_pct": exit_plan.stop_loss_pct,
                        "timestamp": time.time()
                    })
                )
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                import traceback
                traceback.print_exc()
        
        if loop_num < 1:
            print("\n⏳ Waiting 3 seconds...")
            time.sleep(3)
    
    print("\n" + "=" * 60)
    print("✅ ExitBrain v3.5 Validation Complete!")
    print("\n📝 Check Redis keys:")
    print("   quantum:exit_decision:*")
    print()

if __name__ == "__main__":
    main()
