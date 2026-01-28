#!/usr/bin/env python3
"""
Ledger Sync Service - Auto-sync ledger from Binance positionRisk
=================================================================

Fetches current positions from Binance testnet futures
Updates Redis ledger (last_known_amt, last_side, updated_at)
Runs every 15 seconds via systemd timer

Author: Quantum Trader Team
Date: 2026-01-26
"""
import os
import sys
import time
import hmac
import hashlib
import urllib.parse
import urllib.request
import json
import redis

# Config
BINANCE_API_KEY = os.getenv("BINANCE_TESTNET_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_TESTNET_API_SECRET", "")
BINANCE_BASE_URL = "https://testnet.binancefuture.com"

ALLOWLIST_STR = os.getenv("LEDGER_SYNC_ALLOWLIST", "BTCUSDT,ETHUSDT,TRXUSDT")
ALLOWLIST = [s.strip() for s in ALLOWLIST_STR.split(",") if s.strip()]

REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

def fetch_positions():
    """Fetch positionRisk from Binance"""
    params = {"timestamp": int(time.time() * 1000)}
    query_string = urllib.parse.urlencode(params)
    signature = hmac.new(
        BINANCE_API_SECRET.encode(),
        query_string.encode(),
        hashlib.sha256
    ).hexdigest()
    params["signature"] = signature
    
    url = f"{BINANCE_BASE_URL}/fapi/v2/positionRisk?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, method="GET")
    req.add_header("X-MBX-APIKEY", BINANCE_API_KEY)
    
    with urllib.request.urlopen(req, timeout=10) as response:
        return json.loads(response.read().decode())

def sync_ledger():
    """Sync ledger for allowlist symbols"""
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=False)
    
    try:
        positions = fetch_positions()
        
        for symbol in ALLOWLIST:
            pos = next((p for p in positions if p["symbol"] == symbol), None)
            
            if pos:
                amt = float(pos.get("positionAmt", 0))
                entry_price = float(pos.get("entryPrice", 0))
                side = "LONG" if amt > 0 else ("SHORT" if amt < 0 else "FLAT")
                abs_amt = abs(amt)
                
                ledger_key = f"quantum:position:ledger:{symbol}"
                r.hset(ledger_key, mapping={
                    "last_known_amt": str(abs_amt),
                    "last_side": side,
                    "qty": str(abs_amt),
                    "side": side,
                    "entry_price": str(entry_price),
                    "updated_at": str(int(time.time())),
                    "synced_at": str(int(time.time()))
                })
                r.expire(ledger_key, 86400)  # 24h TTL
                
                print(f"✅ {symbol}: {side} {abs_amt:.4f} @ {entry_price:.6f}")
            else:
                print(f"⚠️  {symbol}: not found in positionRisk")
        
        print(f"📊 Ledger sync complete ({len(ALLOWLIST)} symbols)")
        return 0
        
    except Exception as e:
        print(f"❌ Ledger sync failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(sync_ledger())
