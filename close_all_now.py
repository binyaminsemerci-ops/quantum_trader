#!/usr/bin/env python3
"""Close all open positions immediately"""
import os
import requests
import hmac
import hashlib
import time
from urllib.parse import urlencode

API_KEY = os.getenv("BINANCE_API_KEY", "qy5g2Dxa8JyNzWI29Ub9ZBbaEWAbTjTz91OMCOWWBd7LRlrnLSO82uQFLuMWGoHb")
API_SECRET = os.getenv("BINANCE_API_SECRET", "oIcNvT3IzlnDwrPxor8jxmCnt2f7ClmmreLsfog2R1QidbgBXNCvm0KyWcF1ebbg")
BASE_URL = "https://fapi.binance.com"

def sign_request(params):
    query_string = urlencode(params)
    signature = hmac.new(API_SECRET.encode(), query_string.encode(), hashlib.sha256).hexdigest()
    return f"{query_string}&signature={signature}"

def get_positions():
    endpoint = "/fapi/v2/positionRisk"
    params = {"timestamp": int(time.time() * 1000)}
    signed = sign_request(params)
    headers = {"X-MBX-APIKEY": API_KEY}
    
    resp = requests.get(f"{BASE_URL}{endpoint}?{signed}", headers=headers)
    return resp.json()

def close_position(symbol, side, qty):
    endpoint = "/fapi/v1/order"
    params = {
        "symbol": symbol,
        "side": side,
        "type": "MARKET",
        "quantity": qty,
        "reduceOnly": "true",
        "timestamp": int(time.time() * 1000)
    }
    signed = sign_request(params)
    headers = {"X-MBX-APIKEY": API_KEY}
    
    resp = requests.post(f"{BASE_URL}{endpoint}?{signed}", headers=headers)
    return resp.json()

if __name__ == "__main__":
    print("[TARGET] Closing all positions...\n")
    
    positions = get_positions()
    closed = 0
    failed = 0
    
    for pos in positions:
        amt = float(pos['positionAmt'])
        if amt != 0:
            symbol = pos['symbol']
            side = 'SELL' if amt > 0 else 'BUY'
            qty = abs(amt)
            
            print(f"Closing {symbol}: {side} {qty}")
            try:
                result = close_position(symbol, side, qty)
                if 'orderId' in result:
                    closed += 1
                    print(f"[OK] {symbol} closed")
                else:
                    failed += 1
                    print(f"❌ {symbol} failed: {result.get('msg', 'Unknown error')}")
            except Exception as e:
                failed += 1
                print(f"❌ {symbol} error: {e}")
    
    print(f"\n[CHART] Results: [OK] {closed} closed | ❌ {failed} failed")
