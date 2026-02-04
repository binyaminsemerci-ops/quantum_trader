#!/usr/bin/env python3
"""Test all Binance API keys to find which one matches user's UI"""
import urllib.request
import urllib.parse
import json
import hmac
import hashlib
import time

# All API keys found on the VPS
keys = {
    "Key1_DEFAULT": (
        "e9ZqWhGhAEhDPfNBfQMiJv8zULKJZBIwaaJdfbbUQ8ZNj1WUMumrjenHoRzpzUPD",
        "ZowBZEfL1R1ValcYLkbxjMfZ1tOxfEDRW4eloWRGGjk5etn0vSFFSU3gCTdCFoja"
    ),
    "Key2_UNAUTHORIZED": (
        "IsY3mFpko7Z8joZr8clWwpJZuZcFdAtnDBy4g4ULQu827Gf6kJPAPS9VyVOVrR0r",
        "tEKYWf77tqSOArfgeSqgVwiO62bjro8D1VMaEvXBwcUOQuRxmCdxrtTvAXy7ZKSE"
    ),
    "Key3_WORKING": (
        "w2W60kzuCfPJKGIqSvmp0pqISUO8XKICjc5sD8QyJuJpp9LKQgvXKhtd09Ii3rwg",
        "kQvj9TAxdlWc4TZz2ybGe6cLc4nNlHOw7pEAMBUUDQTe5KOPnRU7aCq7nj0rVGvC"
    ),
    "Key4_BACKUP": (
        "xOPqaf2iSKt4gVuScoebb3wDBm0R9gw0qSPtpHYnJNzcahTSL58b4QZcC4dsJ5eX",
        "hwyeOL1BHBMv5jLmCEemg2OQNUb8dUAyHgamOftcS9oFDfc605SX1IZs294zvNmZ"
    )
}

print("=" * 80)
print("TESTING ALL BINANCE API KEYS")
print("=" * 80)
print("\nUser's screenshot shows:")
print("  ARCUSDT:  110,153 qty @ 0.0737245 entry → +178.58 USDT (+43.98%)")
print("  RIVERUSDT: -606.7 qty @ 12.406638 entry → +58.74 USDT (+15.60%)")
print("  ANKRUSDT: 214,300 qty @ 0.005599 entry → +8.91 USDT (+7.43%)")
print("  HYPEUSDT: -252.30 qty @ 33.30872 entry → +96.46 USDT (+5.73%)")
print("  FHEUSDT:  1,675 qty @ 0.1199800 entry → -0.75 USDT (-7.50%)")
print("\n" + "=" * 80)

for name, (api_key, api_secret) in keys.items():
    print(f"\n\n🔑 Testing {name}")
    print(f"   API Key: {api_key[:30]}...")
    
    try:
        timestamp = int(time.time() * 1000)
        params = f"timestamp={timestamp}&recvWindow=5000"
        signature = hmac.new(
            api_secret.encode(),
            params.encode(),
            hashlib.sha256
        ).hexdigest()
        
        url = f"https://testnet.binancefuture.com/fapi/v2/positionRisk?{params}&signature={signature}"
        req = urllib.request.Request(url, headers={"X-MBX-APIKEY": api_key})
        
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read())
            open_positions = [p for p in data if float(p.get("positionAmt", 0)) != 0]
            
            if open_positions:
                print(f"\n   ✅ SUCCESS: Found {len(open_positions)} open positions\n")
                
                # Check if this matches user's screenshot
                matches = []
                for p in open_positions:
                    symbol = p["symbol"]
                    qty = float(p["positionAmt"])
                    entry = float(p["entryPrice"])
                    pnl = float(p["unRealizedProfit"])
                    
                    # Check for specific matches from screenshot
                    if symbol == "ARCUSDT" and abs(qty - 110153) < 1000:
                        matches.append(f"✨ ARCUSDT MATCH! qty={qty:.2f} (expected ~110,153)")
                    elif symbol == "RIVERUSDT" and abs(abs(qty) - 606.7) < 100:
                        matches.append(f"✨ RIVERUSDT MATCH! qty={qty:.2f} (expected ~-606.7)")
                    elif symbol == "ANKRUSDT" and abs(qty - 214300) < 10000:
                        matches.append(f"✨ ANKRUSDT MATCH! qty={qty:.2f} (expected ~214,300)")
                    elif symbol == "HYPEUSDT" and abs(abs(qty) - 252.30) < 50:
                        matches.append(f"✨ HYPEUSDT MATCH! qty={qty:.2f} (expected ~-252.30)")
                    
                    print(f"   {symbol:12} qty={qty:>14,.2f}  entry={entry:>12.6f}  pnl={pnl:>10.2f} USDT")
                
                if matches:
                    print("\n" + "=" * 80)
                    print("🎯 FOUND THE CORRECT KEY! This matches user's UI:")
                    for match in matches:
                        print(f"   {match}")
                    print("=" * 80)
                else:
                    print("\n   ⚠️  No quantity matches - this is NOT the key from the screenshot")
            else:
                print(f"\n   ⚠️  No open positions")
                
    except urllib.error.HTTPError as e:
        error_body = e.read().decode() if e.code == 401 else str(e)
        print(f"\n   ❌ HTTP ERROR {e.code}: {error_body[:100]}")
    except Exception as e:
        print(f"\n   ❌ ERROR: {type(e).__name__}: {str(e)[:100]}")

print("\n\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("\nLook for the key marked with 🎯 - that's the one matching the UI screenshot!")
print("That key should be updated in ALL /etc/quantum/*.env files.\n")
