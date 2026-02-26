"""Test evaluate() on DOGEUSDT and print full traceback"""
import sys, traceback
sys.path.insert(0, "/home/qt/quantum_trader")
sys.path.insert(0, "/home/qt/quantum_trader/microservices/harvest_brain")

import redis
r = redis.Redis(host="localhost", port=6379, decode_responses=True)

# Read real position data
d = r.hgetall("quantum:position:DOGEUSDT")
print("Position hash:", d)

import importlib
import harvest_brain as hb
importlib.reload(hb)

cfg = hb.Config()
policy = hb.HarvestPolicy(cfg, r)

pos = hb.Position(
    symbol="DOGEUSDT",
    side=d.get("side", "SHORT"),
    qty=float(d.get("quantity", 974)),
    entry_price=float(d.get("entry_price", 0.10267)),
    current_price=0.10090,
    unrealized_pnl=(0.10267 - 0.10090) * 974,
    entry_risk=float(d.get("entry_risk_usdt", 3.0)),
    stop_loss=float(d.get("stop_loss", 0.10575)),
    take_profit=float(d.get("take_profit") or 0.09754),
    leverage=float(d.get("leverage", 50)),
    last_update_ts=0,
)
print(f"Testing: {pos.symbol} side={pos.side} qty={pos.qty} pnl={pos.unrealized_pnl:.4f}")
print(f"SL={pos.stop_loss} leverage={pos.leverage}")

try:
    intents = policy.evaluate(pos)
    if intents:
        for i in intents:
            print(f"INTENT: {i.intent_type} qty={i.qty}")
    else:
        print("No intents returned")
except Exception as e:
    print("EXCEPTION:", e)
    traceback.print_exc()
