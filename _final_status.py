#!/usr/bin/env python3
"""Final status check: execution silence, Binance errors, service health"""
import redis, time, subprocess

r = redis.Redis()

# ─── 1. execution.result status ──────────────────────────────────────────
info = r.xinfo_stream("quantum:stream:execution.result")
last = r.xrevrange("quantum:stream:execution.result", count=1)
print("=== execution.result ===")
print(f"  Length: {info['length']}")
if last:
    ts_id = last[0][0].decode()
    ts_ms = int(ts_id.split('-')[0])
    age_h = (int(time.time() * 1000) - ts_ms) // 3600000
    print(f"  Last entry: {ts_id} ({age_h}h ago)")
    for k, v in last[0][1].items():
        print(f"    {k.decode()}: {v.decode()[:80]}")

# ─── 2. Binance errors analysis ───────────────────────────────────────────
entries = r.xrevrange("quantum:stream:apply.result", count=500)

print("\n=== Apply.result Binance error deep dive ===")
exec_errors = {}
for mid, d in entries:
    err = d.get(b'error', b'').decode()
    if 'Binance API error' in err or 'execution_failed' in err:
        symbol = d.get(b'symbol', b'?').decode()
        action_val = d.get(b'action', b'?').decode()
        # Extract Binance error code
        import re
        code_match = re.search(r'"code"\s*:\s*(-?\d+)', err)
        code = code_match.group(1) if code_match else 'unknown'
        msg_match = re.search(r'"msg"\s*:\s*"([^"]+)"', err)
        msg = msg_match.group(1)[:50] if msg_match else err[:50]
        key = (symbol, code)
        if key not in exec_errors:
            exec_errors[key] = {'code': code, 'msg': msg, 'action': action_val, 'count': 0}
        exec_errors[key]['count'] += 1

for (sym, code), info_d in sorted(exec_errors.items(), key=lambda x: -x[1]['count']):
    print(f"  {sym} code={code} ({info_d['count']}x): {info_d['msg']}")

# ─── 3. Apply.result decision=EXECUTE check ──────────────────────────────
print("\n=== apply.result: executing intent count (last 2000) ===")
entries2k = r.xrevrange("quantum:stream:apply.result", count=2000)
decisions = {}
for _, d in entries2k:
    dec = d.get(b'decision', b'absent').decode()
    decisions[dec] = decisions.get(dec, 0) + 1
for k, v in sorted(decisions.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v}")

# ─── 4. Check apply.result for any EXECUTE entries (opened orders) ────────
execute_entries = [(mid, d) for mid, d in entries2k if d.get(b'decision') == b'EXECUTE']
print(f"\n  decision=EXECUTE entries (last 2000): {len(execute_entries)}")

# ─── 5. execution_service log check ──────────────────────────────────────
print("\n=== execution.log last 30 lines ===")
log = subprocess.run(
    ["tail", "-n", "30", "/var/log/quantum/execution.log"],
    capture_output=True, text=True
).stdout
for line in log.splitlines()[-15:]:
    stripped = line.strip()
    if not stripped or "RX apply.result" in stripped:
        continue
    print(f"  {stripped[-180:]}")

# ─── 6. Service status ────────────────────────────────────────────────────
print("\n=== Service statuses ===")
services = [
    "quantum-execution", "quantum-ai-engine", "quantum-intent-executor",
    "quantum-apply-layer", "quantum-monitoring"
]
for svc in services:
    status = subprocess.run(
        ["systemctl", "is-active", svc],
        capture_output=True, text=True
    ).stdout.strip()
    print(f"  {svc}: {status}")

# ─── 7. Check what's blocking intent-executor (why no EXECUTE decisions) ─
print("\n=== intent-executor logs (last 20 relevant lines) ===")
result = subprocess.run(
    ["journalctl", "-u", "quantum-intent-executor", "-n", "50", "--no-pager"],
    capture_output=True, text=True
)
if result.returncode == 0:
    for line in result.stdout.splitlines():
        if any(k in line for k in ["SKIP", "BLOCK", "EXECUTE", "reject", "manual", "drawdown", "guard"]):
            print(f"  {line.strip()[-160:]}")
else:
    # Try log file
    result2 = subprocess.run(
        ["tail", "-n", "50", "/var/log/quantum/intent_executor.log"],
        capture_output=True, text=True
    )
    for line in result2.stdout.splitlines()[-20:]:
        print(f"  {line.strip()[-160:]}")
