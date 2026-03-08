"""Inject an invalid intent (wrong source) and verify it is rejected, not leaked."""
import redis
import time
import uuid

r = redis.Redis(host="127.0.0.1", port=6379, decode_responses=True)

intent_id = "REJECT-TEST-" + str(uuid.uuid4())[:8]
fields = {
    "intent_id": intent_id,
    "symbol": "ETHUSDT",
    "action": "FULL_CLOSE",
    "urgency": "HIGH",
    "side": "LONG",
    "qty_fraction": "1.0",
    "quantity": "0.01",
    "entry_price": "3000.0",
    "mark_price": "3100.0",
    "R_net": "2.0",
    "confidence": "0.90",
    "reason": "reject_path_test",
    "loop_id": "test-loop-002",
    "source": "MALICIOUS_SOURCE",   # <-- V8 should reject this
    "patch": "PATCH-5A",
    "ts_epoch": str(time.time()),
}

before_trade = r.xlen("quantum:stream:trade.intent")
before_apply = r.xlen("quantum:stream:apply.plan")
before_rejected = r.xlen("quantum:stream:exit.intent.rejected")

msg_id = r.xadd("quantum:stream:exit.intent", fields)
print(f"INJECTED BAD intent_id={intent_id} msg_id={msg_id}")

import time as _t
_t.sleep(3)

after_trade = r.xlen("quantum:stream:trade.intent")
after_apply = r.xlen("quantum:stream:apply.plan")
after_rejected = r.xlen("quantum:stream:exit.intent.rejected")

print(f"trade.intent   before={before_trade} after={after_trade} delta={after_trade - before_trade}")
print(f"apply.plan     before={before_apply} after={after_apply} delta={after_apply - before_apply}")
print(f"exit.rejected  before={before_rejected} after={after_rejected} delta={after_rejected - before_rejected}")

if after_trade > before_trade:
    print(f"FAIL: trade.intent received {after_trade - before_trade} message(s) — bad intent leaked!")
else:
    print("PASS: trade.intent unchanged — bad intent correctly blocked")

if after_apply > before_apply:
    print(f"FAIL: apply.plan grew — bad intent reached live path!")
else:
    print("PASS: apply.plan unchanged — no live-path leak")

if after_rejected > before_rejected:
    print(f"PASS: exit.intent.rejected received {after_rejected - before_rejected} reject record(s)")
else:
    print("WARN: exit.intent.rejected did not grow — check gateway log")

# Show the reject record
recent_rej = r.xrevrange("quantum:stream:exit.intent.rejected", "+", "-", count=1)
if recent_rej:
    mid, flds = recent_rej[0]
    print(f"reject record: intent_id={flds.get('intent_id')} rule={flds.get('rule')} reason={flds.get('reason')}")
