import redis, sys

r = redis.Redis(host="localhost", port=6379, decode_responses=True)
keys = sorted([
    k for k in r.keys("quantum:position:*")
    if ":snapshot:" not in k and ":ledger:" not in k and ":claim:" not in k
])

print(f"OPEN POSITIONS — {len(keys)} symbols")
print("-" * 105)
print(f"{'SYMBOL':<24} {'SIDE':<5} {'QTY':<14} {'ENTRY':<14} {'SL':<14} {'TP':<14} {'UNREAL PNL':<12} {'LEV':<5}")
print("-" * 105)

total_pnl = 0.0
for key in keys:
    d = r.hgetall(key)
    sym  = key.replace("quantum:position:", "")
    side = d.get("side", "?")
    qty  = d.get("quantity", "?")
    entry= d.get("entry_price", "?")
    sl   = d.get("stop_loss", "?")
    tp   = d.get("take_profit", "?")
    pnl  = d.get("unrealized_pnl", "?")
    lev  = d.get("leverage", "?")
    try:
        pnl_f = float(pnl)
        total_pnl += pnl_f
        flag = "  <<LOSS>>" if pnl_f < -0.5 else ("  +profit" if pnl_f > 0.5 else "")
        pnl_disp = f"{pnl_f:+.4f}"
    except:
        pnl_disp = pnl
        flag = ""
    print(f"{sym:<24} {side:<5} {qty:<14} {entry:<14} {sl:<14} {tp:<14} {pnl_disp:<12} {lev:<5}{flag}")

print("-" * 105)
print(f"TOTAL UNREALIZED PNL: {total_pnl:+.4f} USDT  ({len(keys)} positions)")
