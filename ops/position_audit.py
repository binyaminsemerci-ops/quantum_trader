#!/usr/bin/env python3
"""
Full position audit: Binance testnet vs Redis vs logs.
Checks position size, PnL, logging gaps.
"""
import sys, os, json, time, hmac, hashlib
sys.path.insert(0, '/home/qt/quantum_trader')
import redis, requests

r = redis.Redis()

# ── 1. Redis positions ──────────────────────────────────────────────
print("=" * 60)
print("REDIS POSITIONS")
print("=" * 60)
pos_keys = [k.decode() for k in r.keys("quantum:position:*")
            if not any(x in k.decode() for x in
                       ["ledger","snapshot","cooldown","dedupe","hold","lock","stream","claim"])]
print(f"Active: {len(pos_keys)}/10\n")
for key in sorted(pos_keys):
    data = {k.decode(): v.decode() for k, v in r.hgetall(key).items()}
    sym = data.get("symbol", key.split(":")[-1])
    side = data.get("side", "?")
    qty = data.get("quantity", data.get("qty", "?"))
    entry = data.get("entry_price", "?")
    lev = data.get("leverage", "?")
    sl = data.get("stop_loss", "?")
    tp = data.get("take_profit", "?")
    pnl = data.get("unrealized_pnl", "missing ⚠️")
    plan_id = data.get("plan_id", "?")[:12]
    src = data.get("source", data.get("source", "apply_layer"))
    created = data.get("created_at")
    age = int(time.time()) - int(created) if created else -1
    risk_missing = data.get("risk_missing", "0")

    print(f"  {sym}")
    print(f"    side={side}  qty={qty}  entry={entry}  lev={lev}x")
    print(f"    SL={sl}  TP={tp}")
    print(f"    unrealized_pnl={pnl}")
    print(f"    plan_id={plan_id}  age={age}s  src={src}")
    if risk_missing == "1":
        print(f"    ⚠️  risk_missing=1 (ATR not available at entry)")
    print()

# ── 2. Binance testnet via direct REST (HMAC) ──────────────────────
print("=" * 60)
print("BINANCE TESTNET POSITIONS")
print("=" * 60)
try:
    API_KEY    = "w2W60kzuCfPJKGIqSvmp0pqISUO8XKICjc5sD8QyJuJpp9LKQgvXKhtd09Ii3rwg"
    API_SECRET = "QI18cg4zcbApc9uaDL8ZUmoAJQthQczZ9cKzORlSJfnK2zBEdLvSLb5ZEgZ6R1Kg"
    BASE_URL   = "https://testnet.binancefuture.com"

    params = {"timestamp": int(time.time() * 1000)}
    query  = "&".join(f"{k}={v}" for k, v in params.items())
    sig    = hmac.new(API_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
    url    = f"{BASE_URL}/fapi/v2/positionRisk?{query}&signature={sig}"
    resp   = requests.get(url, headers={"X-MBX-APIKEY": API_KEY}, timeout=10)
    raw    = resp.json()

    if isinstance(raw, dict):
        print(f"  API error: {raw}")
        raise RuntimeError(f"Binance error: {raw}")

    all_pos = raw

    open_pos = [p for p in all_pos if abs(float(p.get("positionAmt", 0))) > 0]
    print(f"  Open positions on exchange: {len(open_pos)}\n")

    for p in sorted(open_pos, key=lambda x: x["symbol"]):
        sym  = p["symbol"]
        amt  = float(p["positionAmt"])
        entry = float(p["entryPrice"])
        pnl  = float(p.get("unRealizedProfit", p.get("unrealizedProfit", 0)))
        mark = float(p["markPrice"])
        liq  = float(p.get("liquidationPrice", 0))
        lev  = p.get("leverage", "?")
        side = "LONG" if amt > 0 else "SHORT"
        notional = abs(amt) * entry
        pnl_pct  = (pnl / notional * 100) if notional > 0 else 0

        print(f"  {sym}: {side} qty={abs(amt):.4f}  entry={entry:.6f}  mark={mark:.6f}")
        print(f"         lev={lev}x  PnL={pnl:+.4f} USDT ({pnl_pct:+.2f}%)  liq={liq:.6f}")

        redis_key = f"quantum:position:{sym}"
        redis_pos = r.hgetall(redis_key)
        if redis_pos:
            rqty   = float((redis_pos.get(b"quantity") or b"0").decode())
            rentry = float((redis_pos.get(b"entry_price") or b"0").decode())
            rpnl   = redis_pos.get(b"unrealized_pnl", b"missing").decode()
            qty_ok = abs(abs(amt) - abs(rqty)) < 0.01
            print(f"         Redis: qty={rqty}  entry={rentry}  pnl_stored={rpnl}")
            print(f"         Sync : {'✅ qty match' if qty_ok else '⚠️ QTY MISMATCH'}  "
                  f"live_pnl={pnl:+.4f}  redis_pnl={rpnl}")
        else:
            print(f"         Redis: ⚠️  NO quantum:position:{sym} — phantom on exchange!")
        print()

except Exception as e:
    print(f"  ERROR: {e}")
    import traceback; traceback.print_exc()

# ── 3. Apply result stream (log quality check) ─────────────────────
print("=" * 60)
print("RECENT APPLY RESULTS (last 10)")
print("=" * 60)
try:
    entries = r.xrevrange("quantum:stream:apply.result", count=10)
    for eid, fields in entries:
        data = {k.decode(): v.decode() for k, v in fields.items()}
        sym = data.get("symbol", "?")
        plan_id = data.get("plan_id", "?")[:12]
        executed = data.get("executed", "?")
        error = data.get("error", "")
        ts = int(data.get("timestamp", 0))
        age = int(time.time()) - ts
        filled = data.get("filled_qty", "")
        order_id = data.get("order_id", "")
        print(f"  {age:5d}s ago  {sym:12s} executed={executed:5s} error={error[:40]}  {f'filled={filled} order={order_id}' if filled else ''}")
except Exception as e:
    print(f"  ERROR: {e}")

# ── 4. Log field coverage check ────────────────────────────────────
print()
print("=" * 60)
print("LOGGING FIELD COVERAGE CHECK")
print("=" * 60)
required_fields = ["symbol", "side", "quantity", "entry_price", "leverage",
                   "stop_loss", "take_profit", "unrealized_pnl", "plan_id", "created_at"]
for key in sorted(pos_keys):
    data = {k.decode(): v.decode() for k, v in r.hgetall(key).items()}
    sym = data.get("symbol", key.split(":")[-1])
    missing = [f for f in required_fields if not data.get(f)]
    if missing:
        print(f"  {sym}: ⚠️  MISSING fields: {missing}")
    else:
        print(f"  {sym}: ✅ all required fields present")
