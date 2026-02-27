import redis, json, time

r = redis.Redis(host="localhost", port=6379, decode_responses=True)

# Read all open positions from snapshots
snap_keys = r.keys("quantum:position:snapshot:*")
populated = 0

for sk in snap_keys:
    data = r.hgetall(sk)
    amt = float(data.get("position_amt", 0))
    if abs(amt) < 0.0001:
        continue
    
    symbol = sk.replace("quantum:position:snapshot:", "")
    side_str = data.get("side", "NONE")
    entry = float(data.get("entry_price", 0))
    mark = float(data.get("mark_price", entry or 1))
    leverage = int(data.get("leverage", 10))
    qty = abs(amt)
    
    if entry <= 0:
        entry = mark
    if mark <= 0:
        continue

    is_short = side_str == "SHORT" or amt < 0

    # 3% SL, 5% TP from entry (reasonable defaults)
    if is_short:
        sl = round(entry * 1.03, 8)   # SHORT SL: above entry
        tp = round(entry * 0.95, 8)   # SHORT TP: below entry
        pos_side = "SHORT"
    else:
        sl = round(entry * 0.97, 8)   # LONG SL: below entry
        tp = round(entry * 1.05, 8)   # LONG TP: above entry
        pos_side = "LONG"

    risk_usdt = abs(entry - sl) * qty
    
    # Check if position hash already exists with valid SL
    pos_key = f"quantum:position:{symbol}"
    existing = r.hgetall(pos_key)
    existing_sl = float(existing.get("stop_loss", 0))
    
    if existing_sl > 0:
        print(f"  {symbol}: already has SL={existing_sl} — skipping")
        continue

    pos_hash = {
        "symbol": symbol,
        "side": pos_side,
        "quantity": str(qty),
        "entry_price": str(entry),
        "stop_loss": str(sl),
        "take_profit": str(tp),
        "leverage": str(leverage),
        "entry_risk_usdt": str(round(risk_usdt, 6)),
        "atr_value": "0.02",
        "volatility_factor": "1.0",
        "risk_missing": "0",
        "opened_at": str(int(time.time())),
        "source": "populate_existing_positions",
    }
    r.hset(pos_key, mapping=pos_hash)
    print(f"  {symbol}: {pos_side} qty={qty} entry={entry:.6f} SL={sl:.6f} TP={tp:.6f} risk=${risk_usdt:.2f}")
    populated += 1

print(f"\nPopulated {populated} existing positions with SL/TP")
