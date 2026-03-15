#!/usr/bin/env python3
"""
Sync Redis positions to match actual Binance testnet state.
Fixes: qty mismatch, stale PnL, stale leverage.
"""
import sys, time, hmac, hashlib
import redis, requests

r = redis.Redis()

API_KEY    = "w2W60kzuCfPJKGIqSvmp0pqISUO8XKICjc5sD8QyJuJpp9LKQgvXKhtd09Ii3rwg"
API_SECRET = "QI18cg4zcbApc9uaDL8ZUmoAJQthQczZ9cKzORlSJfnK2zBEdLvSLb5ZEgZ6R1Kg"
BASE_URL   = "https://testnet.binancefuture.com"

def binance_positions():
    params = {"timestamp": int(time.time() * 1000)}
    query  = "&".join(f"{k}={v}" for k, v in params.items())
    sig    = hmac.new(API_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
    url    = f"{BASE_URL}/fapi/v2/positionRisk?{query}&signature={sig}"
    resp   = requests.get(url, headers={"X-MBX-APIKEY": API_KEY}, timeout=10)
    raw    = resp.json()
    if isinstance(raw, dict):
        raise RuntimeError(f"Binance API error: {raw}")
    return {p["symbol"]: p for p in raw if abs(float(p.get("positionAmt", 0))) > 0}

def get_redis_positions():
    keys = [k.decode() for k in r.keys("quantum:state:positions:*")]
    return {k.split(":")[-1]: {kk.decode(): vv.decode() for kk, vv in r.hgetall(k).items()}
            for k in keys}

print("=" * 60)
print("SYNC REDIS → BINANCE TESTNET")
print("=" * 60)

bnb = binance_positions()
rds = get_redis_positions()

print(f"\nBinance open: {list(bnb.keys())}")
print(f"Redis  open: {list(rds.keys())}\n")

for sym, bp in bnb.items():
    rkey = f"quantum:state:positions:{sym}"
    rp   = rds.get(sym)

    b_qty  = abs(float(bp["positionAmt"]))
    b_side = "SHORT" if float(bp["positionAmt"]) < 0 else "LONG"
    b_entry = float(bp["entryPrice"])
    b_pnl   = float(bp.get("unRealizedProfit", 0))
    b_mark  = float(bp["markPrice"])
    b_lev   = bp.get("leverage", "?")
    b_liq   = float(bp.get("liquidationPrice", 0))

    if rp:
        r_qty  = float(rp.get("quantity", 0))
        r_entry = float(rp.get("entry_price", 0))
        r_pnl   = rp.get("unrealized_pnl", "missing")
        r_lev   = rp.get("leverage", "?")

        updates = {}
        if abs(b_qty - abs(r_qty)) > 0.001:
            print(f"  {sym}: qty {r_qty} → {b_qty}  ⚠️ FIXING")
            updates["quantity"] = str(b_qty)
        if abs(b_entry - r_entry) > 0.000001:
            updates["entry_price"] = str(b_entry)
        if str(b_lev) != str(r_lev):
            print(f"  {sym}: leverage {r_lev}x → {b_lev}x  ⚠️ FIXING")
            updates["leverage"] = str(b_lev)

        # Always update live PnL, mark price, liquidation price
        updates["unrealized_pnl"]  = str(b_pnl)
        updates["mark_price"]      = str(b_mark)
        updates["liquidation_price"] = str(b_liq)
        updates["pnl_updated_at"]  = str(int(time.time()))

        r.hset(rkey, mapping=updates)
        print(f"  {sym}: ✅ synced  qty={b_qty}  lev={b_lev}x  "
              f"live_pnl={b_pnl:+.4f} USDT  liq={b_liq:.6f}")
    else:
        print(f"  {sym}: ⚠️  on Binance but NOT in Redis — writing new entry")
        r.hset(rkey, mapping={
            "symbol":       sym,
            "side":         b_side,
            "position_amt": str(float(bp["positionAmt"])),
            "quantity":     str(b_qty),
            "entry_price":  str(b_entry),
            "leverage":     str(b_lev),
            "unrealized_pnl": str(b_pnl),
            "mark_price":   str(b_mark),
            "current_price": str(b_mark),
            "liquidation_price": str(b_liq),
            "ts_epoch":     str(int(time.time())),
            "source":       "sync_script",
        })
        print(f"  {sym}: ✅ created  qty={b_qty}  lev={b_lev}x  pnl={b_pnl:+.4f}")

# Remove Redis positions not on Binance
for sym in list(rds.keys()):
    if sym not in bnb:
        rkey = f"quantum:state:positions:{sym}"
        r.delete(rkey)
        print(f"\n  {sym}: ❌ in Redis but NOT on Binance — DELETED from Redis (phantom)")

# Final state
print("\n" + "=" * 60)
print("FINAL STATE AFTER SYNC")
print("=" * 60)
rds2 = get_redis_positions()
bnb2 = binance_positions()
for sym in sorted(set(list(rds2.keys()) + list(bnb2.keys()))):
    rp = rds2.get(sym, {})
    bp = bnb2.get(sym, {})
    r_qty = rp.get("quantity", "—")
    b_qty = abs(float(bp.get("positionAmt", 0))) if bp else "—"
    r_pnl = rp.get("unrealized_pnl", "—")
    b_pnl = float(bp.get("unRealizedProfit", 0)) if bp else "—"
    r_lev = rp.get("leverage", "—")
    print(f"  {sym:14s}  qty_redis={r_qty:10}  qty_bnb={b_qty}  "
          f"lev={r_lev}x  pnl={b_pnl:+.4f}")
