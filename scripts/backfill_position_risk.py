#!/usr/bin/env python3
"""
Backfill script: Populate risk fields for positions missing them.
Scans quantum:position:* for missing entry_risk_usdt and backfills from
quantum:stream:trade.intent history (last ENTRY intent per symbol).
"""

import redis
import time
from typing import Optional, Dict, Any

# Constants
REDIS_HOST = "localhost"
REDIS_PORT = 6379
TRADE_INTENT_STREAM = "quantum:stream:trade.intent"

# Risk computation
def compute_risk_fields(atr_value: float, volatility_factor: float, entry_price: float, quantity: float) -> Dict[str, float]:
    """Compute risk fields from ATR and volatility."""
    risk_price = atr_value * volatility_factor
    entry_risk_usdt = abs(quantity) * risk_price
    
    return {
        'atr_value': atr_value,
        'volatility_factor': volatility_factor,
        'risk_price': risk_price,
        'entry_risk_usdt': entry_risk_usdt,
        'risk_missing': 0
    }

def find_last_entry_intent(r: redis.Redis, symbol: str) -> Optional[Dict[str, Any]]:
    """Find the most recent ENTRY intent for a symbol from trade.intent stream."""
    try:
        # Use XREVRANGE to scan backwards from latest
        entries = r.xrevrange(TRADE_INTENT_STREAM, count=500)
        
        for entry_id, fields in entries:
            # Check if this is ENTRY for our symbol
            action = fields.get('action', '')
            entry_symbol = fields.get('symbol', '')
            
            if action == 'ENTRY' and entry_symbol == symbol:
                # Extract fields
                atr_value = float(fields.get('atr_value', 0.0))
                volatility_factor = float(fields.get('volatility_factor', 0.0))
                entry_price = float(fields.get('entry_price', 0.0))
                quantity = float(fields.get('quantity', 0.0))
                
                if atr_value > 0 and volatility_factor > 0 and entry_price > 0:
                    return {
                        'atr_value': atr_value,
                        'volatility_factor': volatility_factor,
                        'entry_price': entry_price,
                        'quantity': quantity,
                        'stream_id': entry_id
                    }
        
        return None
    
    except Exception as e:
        print(f"Error finding intent for {symbol}: {e}")
        return None

def backfill_position(r: redis.Redis, symbol: str) -> bool:
    """Backfill risk fields for a single position. Returns True if updated."""
    pos_key = f"quantum:position:{symbol}"
    
    try:
        # Get position data
        pos_data = r.hgetall(pos_key)
        if not pos_data:
            print(f"[SKIP] {symbol}: Position key not found")
            return False
        
        # Check if already has risk fields
        entry_risk_usdt = float(pos_data.get('entry_risk_usdt', 0.0))
        if entry_risk_usdt > 0:
            print(f"[SKIP] {symbol}: Already has entry_risk_usdt={entry_risk_usdt:.2f}")
            return False
        
        # Get quantity and entry price from position
        qty = float(pos_data.get('quantity', 0.0))
        entry_price = float(pos_data.get('entry_price', 0.0))
        
        if qty == 0:
            print(f"[SKIP] {symbol}: Position closed (qty=0)")
            return False
        
        # Find last ENTRY intent
        intent_data = find_last_entry_intent(r, symbol)
        if not intent_data:
            print(f"[WARN] {symbol}: No ENTRY intent found in stream")
            r.hset(pos_key, 'risk_missing', 1)
            return False
        
        # Compute risk fields
        risk_fields = compute_risk_fields(
            intent_data['atr_value'],
            intent_data['volatility_factor'],
            intent_data.get('entry_price', entry_price),  # Fallback to position entry_price
            qty
        )
        
        # Update position
        r.hset(pos_key, mapping=risk_fields)
        
        print(
            f"[OK] {symbol}: "
            f"atr={risk_fields['atr_value']:.4f} "
            f"vol={risk_fields['volatility_factor']:.1f} "
            f"entry_risk={risk_fields['entry_risk_usdt']:.2f} "
            f"(stream={intent_data['stream_id'][:16]}...)"
        )
        return True
    
    except Exception as e:
        print(f"[ERROR] {symbol}: {e}")
        return False

def main():
    """Main backfill routine."""
    print("=" * 70)
    print("Position Risk Backfill Script")
    print("=" * 70)
    print()
    
    # Connect to Redis
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    
    try:
        r.ping()
        print(f"✓ Connected to Redis {REDIS_HOST}:{REDIS_PORT}")
    except Exception as e:
        print(f"✗ Redis connection failed: {e}")
        return
    
    print()
    
    # Scan all position keys
    print("Scanning quantum:position:* keys...")
    position_keys = []
    cursor = 0
    while True:
        cursor, keys = r.scan(cursor=cursor, match='quantum:position:*', count=100)
        # Filter out ledger/snapshot
        position_keys.extend([
            k for k in keys 
            if ':ledger:' not in k and ':snapshot:' not in k
        ])
        if cursor == 0:
            break
    
    print(f"Found {len(position_keys)} position keys")
    print()
    
    # Backfill each position
    updated_count = 0
    skipped_count = 0
    failed_count = 0
    
    for pos_key in sorted(position_keys):
        symbol = pos_key.replace('quantum:position:', '')
        if backfill_position(r, symbol):
            updated_count += 1
        else:
            # Check if it was a real failure or skip
            pos_data = r.hgetall(pos_key)
            if float(pos_data.get('entry_risk_usdt', 0.0)) > 0:
                skipped_count += 1
            else:
                failed_count += 1
    
    print()
    print("=" * 70)
    print(f"Backfill Complete:")
    print(f"  Updated: {updated_count}")
    print(f"  Skipped (already valid): {skipped_count}")
    print(f"  Failed (no intent found): {failed_count}")
    print(f"  Total processed: {len(position_keys)}")
    print("=" * 70)

if __name__ == '__main__':
    main()
