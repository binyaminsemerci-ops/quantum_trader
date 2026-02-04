#!/usr/bin/env python3
"""
Sync ALL position data from Binance API to Redis.
Fixes stale quantity, entry_price, unrealized_pnl.
"""

import os
import redis
import urllib.request
import json
import hmac
import hashlib
import time

# Binance API credentials
API_KEY = os.getenv("BINANCE_TESTNET_API_KEY", "w2W60kzuCfPJKGIqSvmp0pqISUO8XKICjc5sD8QyJuJpp9LKQgvXKhtd09Ii3rwg")
API_SECRET = os.getenv("BINANCE_TESTNET_API_SECRET", "QI18cg4zcbApc9uaDL8ZUmoAJQthQczZ9cKzORlSJfnK2zBEdLvSLb5ZEgZ6R1Kg")
BASE_URL = "https://testnet.binancefuture.com"

def fetch_positions():
    """Fetch all positions from Binance /fapi/v2/positionRisk"""
    timestamp = int(time.time() * 1000)
    query_string = f"timestamp={timestamp}"
    signature = hmac.new(
        API_SECRET.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    url = f"{BASE_URL}/fapi/v2/positionRisk?{query_string}&signature={signature}"
    req = urllib.request.Request(url)
    req.add_header('X-MBX-APIKEY', API_KEY)
    
    with urllib.request.urlopen(req, timeout=10) as response:
        return json.loads(response.read().decode())

def sync_to_redis(r, positions):
    """Update Redis position hashes with actual exchange data"""
    updated = 0
    created = 0
    closed = 0
    
    # Track symbols we've seen on exchange
    exchange_symbols = set()
    
    for pos in positions:
        symbol = pos['symbol']
        position_amt = float(pos.get('positionAmt', 0))
        entry_price = float(pos.get('entryPrice', 0))
        unrealized_pnl = float(pos.get('unRealizedProfit', 0))
        leverage = int(pos.get('leverage', 20))
        
        exchange_symbols.add(symbol)
        pos_key = f"quantum:position:{symbol}"
        
        # Skip if no position on exchange
        if position_amt == 0:
            continue
        
        # Determine side
        side = 'LONG' if position_amt > 0 else 'SHORT'
        
        # Check if position exists in Redis
        existing = r.hgetall(pos_key)
        
        if existing:
            # Update existing position
            old_qty = float(existing.get('quantity', 0))
            old_entry = float(existing.get('entry_price', 0))
            old_pnl = float(existing.get('unrealized_pnl', 0))
            
            updates = {
                'quantity': str(abs(position_amt)),
                'entry_price': str(entry_price),
                'unrealized_pnl': str(unrealized_pnl),
                'side': side,
                'leverage': str(leverage)
            }
            
            r.hset(pos_key, mapping=updates)
            
            print(f"[UPDATE] {symbol:12s} qty: {old_qty:>12.2f} → {abs(position_amt):>12.2f} | "
                  f"entry: {old_entry:.6f} → {entry_price:.6f} | "
                  f"pnl: {old_pnl:>8.2f} → {unrealized_pnl:>8.2f}")
            updated += 1
        else:
            # Create new position
            position_data = {
                'symbol': symbol,
                'side': side,
                'quantity': str(abs(position_amt)),
                'entry_price': str(entry_price),
                'unrealized_pnl': str(unrealized_pnl),
                'leverage': str(leverage),
                'created_at': str(int(time.time())),
                'risk_missing': '1'  # Will be backfilled later
            }
            
            r.hset(pos_key, mapping=position_data)
            
            print(f"[CREATE] {symbol:12s} qty: {abs(position_amt):>12.2f} | "
                  f"entry: {entry_price:.6f} | pnl: {unrealized_pnl:>8.2f}")
            created += 1
    
    # Mark closed positions in Redis
    cursor = 0
    while True:
        cursor, keys = r.scan(cursor=cursor, match='quantum:position:*', count=100)
        for key in keys:
            if ':ledger:' in key or ':snapshot:' in key:
                continue
            
            symbol = key.replace('quantum:position:', '')
            if symbol not in exchange_symbols:
                # Position not on exchange - mark as closed
                qty = float(r.hget(key, 'quantity') or 0)
                if qty > 0:
                    r.hset(key, 'quantity', '0')
                    print(f"[CLOSE]  {symbol:12s} qty: {qty:>12.2f} → 0.00 (not on exchange)")
                    closed += 1
        
        if cursor == 0:
            break
    
    return updated, created, closed

def main():
    print("=" * 80)
    print("Sync Positions from Binance to Redis")
    print("=" * 80)
    print()
    
    # Connect to Redis
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    
    try:
        r.ping()
        print("✓ Connected to Redis")
    except Exception as e:
        print(f"✗ Redis connection failed: {e}")
        return
    
    print()
    
    # Fetch positions from Binance
    print("Fetching positions from Binance testnet...")
    try:
        positions = fetch_positions()
        open_positions = [p for p in positions if float(p.get('positionAmt', 0)) != 0]
        print(f"✓ Found {len(open_positions)} open positions on exchange")
    except Exception as e:
        print(f"✗ Failed to fetch positions: {e}")
        return
    
    print()
    print("Syncing to Redis...")
    print("-" * 80)
    
    updated, created, closed = sync_to_redis(r, positions)
    
    print("-" * 80)
    print()
    print(f"Summary:")
    print(f"  Updated: {updated}")
    print(f"  Created: {created}")
    print(f"  Closed: {closed}")
    print(f"  Total: {updated + created + closed}")
    print()
    print("=" * 80)
    print("✓ Sync complete! Run backfill_position_risk.py to update risk fields.")
    print("=" * 80)

if __name__ == '__main__':
    main()
