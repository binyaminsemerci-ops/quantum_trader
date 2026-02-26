import shutil

filepath = "/home/qt/quantum_trader/microservices/intent_bridge/main.py"
shutil.copy2(filepath, filepath + ".bak_posfilter")

with open(filepath, "r") as f:
    content = f.read()

old = "        # Publish to apply.plan\n        try:\n            self._publish_plan(plan_id, intent)"

new = """        # Gate: skip if same symbol already has an open position (2026-02-25)
        # intent_bridge was publishing SELL entries for already-open SHORTs -> spam
        _bridge_snap_key = f"quantum:position:snapshot:{intent['symbol']}"
        try:
            _bridge_snap_raw = self.redis.hget(_bridge_snap_key, "position_amt")
            _bridge_snap_amt = float(_bridge_snap_raw) if _bridge_snap_raw else 0.0
        except Exception:
            _bridge_snap_amt = 0.0
        if abs(_bridge_snap_amt) > 0.0:
            logger.debug(
                f"[BRIDGE_SKIP] {intent['symbol']}: already_open={_bridge_snap_amt} — skipping new entry"
            )
            self._mark_seen(stream_id_str)
            self.redis.xack(INTENT_STREAM, CONSUMER_GROUP, stream_id)
            return

        # Publish to apply.plan
        try:
            self._publish_plan(plan_id, intent)"""

count = content.count(old)
if count != 1:
    print(f"ERROR: found {count} occurrences (need exactly 1)")
    exit(1)

content = content.replace(old, new)
with open(filepath, "w") as f:
    f.write(content)
print("SUCCESS: position filter added to intent_bridge before _publish_plan")
print("Backup: " + filepath + ".bak_posfilter")
