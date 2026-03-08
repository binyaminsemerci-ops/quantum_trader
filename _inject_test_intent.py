"""Inject a synthetic valid exit intent for staged enablement test."""
import redis
import time
import uuid

r = redis.Redis(host="127.0.0.1", port=6379, decode_responses=True)

intent_id = "TEST-" + str(uuid.uuid4())[:8]
fields = {
    "intent_id": intent_id,
    "symbol": "BTCUSDT",
    "action": "FULL_CLOSE",
    "urgency": "HIGH",
    "side": "LONG",
    "qty_fraction": "1.0",
    "quantity": "0.01",
    "entry_price": "60000.0",
    "mark_price": "63000.0",
    "R_net": "2.5",
    "confidence": "0.85",
    "reason": "staged_enablement_test",
    "loop_id": "test-loop-001",
    "source": "exit_management_agent",
    "patch": "PATCH-5A",
    "ts_epoch": str(time.time()),
}

before_trade = r.xlen("quantum:stream:trade.intent")
before_apply = r.xlen("quantum:stream:apply.plan")
before_exit  = r.xlen("quantum:stream:exit.intent")

msg_id = r.xadd("quantum:stream:exit.intent", fields)
print(f"INJECTED intent_id={intent_id} msg_id={msg_id}")
print(f"exit.intent before={before_exit} after={r.xlen('quantum:stream:exit.intent')}")

import time as _t
_t.sleep(3)

after_trade = r.xlen("quantum:stream:trade.intent")
after_apply = r.xlen("quantum:stream:apply.plan")
after_exit  = r.xlen("quantum:stream:exit.intent")

print(f"trade.intent before={before_trade} after={after_trade} delta={after_trade - before_trade}")
print(f"apply.plan   before={before_apply} after={after_apply} delta={after_apply - before_apply}")

# Check most recent trade.intent entry
recent = r.xrevrange("quantum:stream:trade.intent", "+", "-", count=1)
if recent:
    msg_id_r, flds = recent[0]
    import json
    payload = json.loads(flds.get("payload", "{}"))
    print(f"trade.intent latest: msg_id={msg_id_r} symbol={payload.get('symbol')} side={payload.get('side')} type={payload.get('type')} reduceOnly={payload.get('reduceOnly')}")

# Confirm no new apply.plan entries
if after_apply > before_apply:
    print(f"FAIL: apply.plan grew by {after_apply - before_apply}")
else:
    print("PASS: apply.plan unchanged (no leak)")

if after_trade > before_trade:
    print(f"PASS: trade.intent received {after_trade - before_trade} new message(s) via gateway")
else:
    print("WARN: trade.intent did not grow — check gateway log")
