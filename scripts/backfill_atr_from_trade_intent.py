#!/usr/bin/env python3
"""
Backfill ATR and volatility_factor for existing positions from trade.intent stream
This fixes positions with risk_missing=1
"""
import redis
import json
import sys

def backfill_atr():
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    
    # Get all positions
    positions = r.keys("quantum:position:*")
    print(f"Found {len(positions)} positions\n")
    
    updated = 0
    skipped = 0
    
    for pos_key in positions:
        data = r.hgetall(pos_key)
        symbol = data.get('symbol', pos_key.split(':')[-1])
        atr = float(data.get('atr_value', 0))
        vol = float(data.get('volatility_factor', 0))
        risk_missing = int(data.get('risk_missing', 0))
        
        # Skip if already has ATR
        if atr > 0 and vol > 0 and risk_missing == 0:
            skipped += 1
            continue
        
        print(f"Fixing {symbol}: atr={atr}, vol={vol}, risk_missing={risk_missing}")
        
        # Search trade.intent for this symbol
        found = False
        intent_messages = r.xrevrange('quantum:stream:trade.intent', '+', '-', count=200)
        
        for msg_id, fields in intent_messages:
            intent_data = {}
            for k, v in fields.items():
                key = k.decode() if isinstance(k, bytes) else k
                val = v.decode() if isinstance(v, bytes) else v
                intent_data[key] = val
            
            if intent_data.get('symbol') == symbol:
                intent_atr = float(intent_data.get('atr_value', 0))
                intent_vol = float(intent_data.get('volatility_factor', 0))
                
                if intent_atr > 0 and intent_vol > 0:
                    # Compute risk values
                    entry_price = float(data.get('entry_price', 0))
                    qty = float(data.get('quantity', 0))
                    risk_price = intent_atr * intent_vol
                    entry_risk_usdt = abs(qty) * risk_price
                    
                    # Update position
                    r.hset(pos_key, mapping={
                        'atr_value': str(intent_atr),
                        'volatility_factor': str(intent_vol),
                        'risk_price': str(risk_price),
                        'entry_risk_usdt': str(entry_risk_usdt),
                        'risk_missing': '0'
                    })
                    
                    print(f"  ✅ Updated: atr={intent_atr:.6f}, vol={intent_vol:.4f}, risk_usdt={entry_risk_usdt:.2f}")
                    updated += 1
                    found = True
                    break
        
        if not found:
            # Use conservative default ATR values for old positions
            # This allows Harvest Brain to monitor them for TP/SL
            entry_price = float(data.get('entry_price', 0))
            leverage = float(data.get('leverage', 1.0))
            qty = float(data.get('quantity', 0))
            
            # Conservative ATR: 2% of entry price
            default_atr = entry_price * 0.02
            default_vol = 1.5  # Conservative volatility multiplier
            
            risk_price = default_atr * default_vol
            entry_risk_usdt = abs(qty) * risk_price
            
            # Update with defaults
            r.hset(pos_key, mapping={
                'atr_value': str(default_atr),
                'volatility_factor': str(default_vol),
                'risk_price': str(risk_price),
                'entry_risk_usdt': str(entry_risk_usdt),
                'risk_missing': '0'
            })
            
            print(f"  ⚙️  Set default: atr={default_atr:.6f} (2% of entry), vol={default_vol}, risk_usdt={entry_risk_usdt:.2f}")
            updated += 1
    
    print(f"\n📊 Summary:")
    print(f"   Updated: {updated}")
    print(f"   Skipped (already OK): {skipped}")
    print(f"   Failed: {len(positions) - updated - skipped}")
    
    return updated

if __name__ == '__main__':
    try:
        updated = backfill_atr()
        sys.exit(0 if updated > 0 else 1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
