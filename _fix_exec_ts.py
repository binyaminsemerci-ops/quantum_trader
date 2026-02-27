#!/usr/bin/env python3
"""
1. Show full trade.intent entry payload
2. Fix timestamp parsing in execution_service.py
"""
import redis, json, re, shutil, time

r = redis.Redis(decode_responses=True)

# ─── 1. Show raw trade.intent entries ─────────────────────────────────────
print("=== trade.intent: raw entry structure ===")
recent = r.xrevrange("quantum:stream:trade.intent", count=3)
for sid, data in recent:
    print(f"\n  Stream ID: {sid}")
    raw = data.get("payload", "")
    if raw:
        try:
            p = json.loads(raw)
            for k, v in sorted(p.items()):
                val = str(v)
                if len(val) > 120:
                    val = val[:120] + "..."
                print(f"    {k}: {val}")
        except Exception as e:
            print(f"    (payload parse error: {e})")
            print(f"    raw[:300]: {raw[:300]}")
    else:
        for k, v in sorted(data.items()):
            val = str(v)
            if len(val) > 120:
                val = val[:120] + "..."
            print(f"    {k}: {val}")

# ─── 2. Show execution_service.py lines 1023-1055 (timestamp code) ────────
print("\n=== execution_service.py timestamp section (1023-1055) ===")
SRC = "/home/qt/quantum_trader/services/execution_service.py"
with open(SRC) as f:
    lines = f.readlines()

for i, line in enumerate(lines[1022:1055], 1023):
    print(f"  {i:4}: {line.rstrip()}")

# ─── 3. Fix timestamp parsing ─────────────────────────────────────────────
print("\n=== Applying timestamp fix ===")

with open(SRC) as f:
    content = f.read()

OLD_TS = """            intent_timestamp = signal_data.get('timestamp')
            if intent_timestamp:
                try:
                    intent_time = date_parser.isoparse(intent_timestamp)"""

NEW_TS = """            intent_timestamp = signal_data.get('timestamp')
            if intent_timestamp:
                try:
                    ts_str = str(intent_timestamp).strip()
                    if ts_str.isdigit():
                        # Unix timestamp (ms if len>11, else seconds)
                        ts_sec = int(ts_str) / 1000 if len(ts_str) > 11 else int(ts_str)
                        from datetime import datetime as _dt, timezone as _tz
                        intent_time = _dt.fromtimestamp(ts_sec, tz=_tz.utc)
                    else:
                        intent_time = date_parser.isoparse(ts_str)"""

if OLD_TS.strip() in content:
    backup = SRC + f".bak.ts.{int(time.time())}"
    shutil.copy(SRC, backup)
    print(f"  Backup: {backup}")
    content = content.replace(OLD_TS, NEW_TS)
    with open(SRC, "w") as f:
        f.write(content)
    print(f"  ✅ Timestamp fix applied")
else:
    # Show what the actual code looks like to understand mismatch
    print("  Exact string not found — showing actual lines 1022-1027:")
    for i, line in enumerate(lines[1021:1027], 1022):
        print(f"    {i}: {repr(line.rstrip())}")

# ─── 4. Show execution_service.py TradeIntent parsing (1160-1200) ─────────
print("\n=== execution_service.py TradeIntent parse (1160-1205) ===")
for i, line in enumerate(lines[1159:1205], 1160):
    print(f"  {i:4}: {line.rstrip()}")
