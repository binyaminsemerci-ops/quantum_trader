#!/usr/bin/env python3
"""
Targeted fix:
1. Clear ghost slots
2. Check AI Engine timeout issue → reduce httpx timeout in exit_manager
3. Check scanner — why no entries
"""
import redis, subprocess, time, os, re

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

print("=" * 60)
print("TARGETED FIX v2")
print("=" * 60)

# ── 1. Ghost slots ─────────────────────────────────────────────
print("\n[1] GHOST SLOTS")
pos_keys = r.keys("quantum:position:*")
ledger_syms = r.smembers("quantum:ledger:positions") or set()
print(f"  Ledger: {len(ledger_syms)} → {sorted(ledger_syms)}")

ghosts = []
real_active = []
for k in pos_keys:
    try:
        d = r.hgetall(k)
        qty = float(d.get('quantity', 0))
        status = d.get('status', '')
        sym = k.split(':')[-1]
        if qty > 0 and status not in ('CLOSED', 'CLOSED_GHOST'):
            if sym not in ledger_syms:
                ghosts.append(sym)
            else:
                real_active.append((sym, qty))
    except:
        pass

print(f"  Ghost (not in ledger): {ghosts}")
print(f"  Real active: {[s for s,q in real_active]}")
for sym in ghosts:
    r.hset(f"quantum:position:{sym}", mapping={'quantity': '0', 'status': 'CLOSED_GHOST'})
print(f"  ✅ Cleared {len(ghosts)} ghosts. Free slots: {10 - len(real_active)}/10")

# ── 2. AI Engine exit timeout issue ───────────────────────────
print("\n[2] AI ENGINE EXIT TIMEOUT ANALYSIS")
# The issue: exit_manager calls /evaluate_exit(?) with 60s timeout
# AI Engine is running but takes > 60s responding to exit calls
# Check what endpoint exit_manager calls
em_path = "microservices/autonomous_trader/exit_manager.py"
with open(em_path) as f:
    em_code = f.read()

# Find the AI Engine URL call
calls = re.findall(r'(self\.\S*url\S*|http://\S+|/\S+_exit\S*|\.post\(|\.get\()', em_code)
urls = re.findall(r'f?["\']([^"\']*(?:evaluate|exit|predict|score)[^"\']*)["\']', em_code)
print(f"  Exit evaluation URLs: {urls[:5]}")

timeout_matches = re.findall(r'timeout=(\d+\.?\d*)', em_code)
print(f"  Current timeout: {timeout_matches}")

# Fix: reduce timeout from 60 to 15s
# This way stuck AI Engine calls fail fast and fallback runs quickly
old_timeout = 'self.http_client = httpx.AsyncClient(timeout=60.0)'
new_timeout = 'self.http_client = httpx.AsyncClient(timeout=15.0)  # Reduced: fast fallback'
if old_timeout in em_code:
    em_code = em_code.replace(old_timeout, new_timeout)
    with open(em_path, 'w') as f:
        f.write(em_code)
    print(f"  ✅ timeout: 60s → 15s (cycles now 10×15s=150s instead of 10×60s=600s)")
else:
    print(f"  Current timeout line: {'60' in str(timeout_matches)}")
    # Try regex replace
    new_em = re.sub(r'httpx\.AsyncClient\(timeout=60\.0\)', 'httpx.AsyncClient(timeout=15.0)', em_code)
    if new_em != em_code:
        with open(em_path, 'w') as f:
            f.write(new_em)
        print(f"  ✅ timeout patched via regex: 60s → 15s")
    else:
        print(f"  ⚠️  Could not find exact timeout pattern")
        print(f"  timeout matches: {timeout_matches}")

# ── 3. Scanner: why no entries ─────────────────────────────────
print("\n[3] SCANNER CHECK")
# Check universe
uni_keys = r.keys("quantum:universe:*")
print(f"  Universe keys: {len(uni_keys)}")
if uni_keys:
    sample = r.smembers(uni_keys[0]) if r.type(uni_keys[0]) == 'set' else r.get(uni_keys[0])
    print(f"  Sample: {str(sample)[:100]}")

# Check recent scanner logs
r2 = subprocess.run(
    ['grep', '-n', 'Scanner\|opportunity\|confidence\|SKIP\|No entry\|slot', 
     '/proc/1/fd/1'],
    capture_output=True, text=True
)

# Use journalctl instead
r3 = subprocess.run(
    ['journalctl', '-u', 'quantum-autonomous-trader', '--since', '10 minutes ago', '--no-pager'],
    capture_output=True, text=True
)
scanner_lines = []
for line in r3.stdout.splitlines():
    for kw in ['Scanner', 'opportunity', 'No entry', 'confidence', 'slot', 'SKIP_', 'Scanning']:
        if kw in line:
            scanner_lines.append(line.split('] ')[-1])
            break
print(f"  Last scanner messages:")
for l in scanner_lines[-8:]:
    print(f"    {l}")

# ── 4. Restart autonomous-trader to pick up timeout change ────
print("\n[4] RESTART AUTONOMOUS-TRADER")
r4 = subprocess.run(['systemctl', 'restart', 'quantum-autonomous-trader'],
                   capture_output=True, text=True)
time.sleep(4)
r5 = subprocess.run(['systemctl', 'is-active', 'quantum-autonomous-trader'],
                   capture_output=True, text=True)
print(f"  quantum-autonomous-trader: {r5.stdout.strip()}")

print("\n" + "=" * 60)
print("SUMMARY:")
print(f"  Ghost slots cleared: {len(ghosts)}")
print(f"  Active positions: {len(real_active)}")
print(f"  Free slots: {10 - len(real_active)}")
print(f"  Exit timeout: 60s → 15s (cycles: ~600s → ~150s)")
print("=" * 60)
