#!/usr/bin/env python3
"""
TRUTH CHECK — Prove all fixes are real and system is operational.
Run on VPS: python3 /tmp/_truth.py
"""
import redis
import json
import subprocess
from datetime import datetime

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "

results = []

def check(label, passed, detail=""):
    icon = PASS if passed else FAIL
    results.append((passed, label, detail))
    print(f"{icon} {label}")
    if detail:
        print(f"   {detail}")

print("=" * 60)
print("QUANTUM TRADER — TRUTH CHECK")
print(f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
print("=" * 60)

# ─── 1. INTENT-EXECUTOR: BINANCE_BASE_URL ───────────────────
print("\n── 1. INTENT-EXECUTOR ENV ──")
try:
    with open('/etc/quantum/intent-executor.env') as f:
        env = f.read()
    
    has_testnet_url = 'testnet.binancefuture.com' in env
    has_real_url = 'https://fapi.binance.com' in env
    
    for line in env.splitlines():
        if 'BINANCE_BASE_URL' in line or 'BASE_URL' in line:
            print(f"   {line.strip()}")
    
    check("BINANCE_BASE_URL points to testnet", has_testnet_url,
          "testnet.binancefuture.com" if has_testnet_url else "STILL fapi.binance.com!")
    check("Real fapi.binance.com NOT present", not has_real_url,
          "correctly absent" if not has_real_url else "STILL PRESENT — risk of real trades!")
except Exception as e:
    check("Read intent-executor.env", False, str(e))

# ─── 2. INTENT-EXECUTOR: HARVEST SUCCESS COUNT ──────────────
print("\n── 2. HARVEST EXECUTION METRICS ──")
try:
    result = subprocess.run(
        ['journalctl', '-u', 'quantum-intent-executor', '-n', '500', '--no-pager', '--output=short'],
        capture_output=True, text=True
    )
    logs = result.stdout

    harvest_success = logs.count('HARVEST SUCCESS')
    harvest_fail = logs.count('HTTP Error 401') + logs.count('Unauthorized')
    harvest_skip_no_pos = logs.count('HARVEST SKIP')
    
    print(f"   In last 500 log lines:")
    print(f"   HARVEST SUCCESS:  {harvest_success}")
    print(f"   401 Errors:       {harvest_fail}")
    print(f"   HARVEST SKIP:     {harvest_skip_no_pos}")
    
    check("HARVEST SUCCESS > 0 (executions happening)", harvest_success > 0,
          f"{harvest_success} harvests in recent logs")
    check("401 Errors = 0 (no more unauthorized)", harvest_fail == 0,
          f"{harvest_fail} 401 errors" if harvest_fail > 0 else "clean")
    
    # Get most recent harvest success
    for line in reversed(logs.splitlines()):
        if 'HARVEST SUCCESS' in line:
            print(f"   Most recent: {line.strip()[-80:]}")
            break
except Exception as e:
    check("Read journalctl harvest metrics", False, str(e))

# ─── 3. SLOT COUNT: GHOST POSITIONS CLEARED ─────────────────
print("\n── 3. GHOST POSITIONS / SLOT COUNT ──")
try:
    ghost_count = 0
    open_count = 0
    closed_ghost_count = 0
    
    cursor = 0
    while True:
        cursor, keys = r.scan(cursor, match="quantum:position:*", count=200)
        for key in keys:
            if ":snapshot:" in key or ":ledger:" in key or ":phantom_backup" in key:
                continue
            qty_raw = r.hget(key, "quantity")
            status = r.hget(key, "status") or ""
            
            if status == "CLOSED_GHOST":
                closed_ghost_count += 1
            elif qty_raw is not None:
                try:
                    if abs(float(qty_raw)) > 0:
                        open_count += 1
                except:
                    pass
        if cursor == 0:
            break
    
    print(f"   CLOSED_GHOST hashes (fixed): {closed_ghost_count}")
    print(f"   qty > 0 (real open):         {open_count}")
    
    check("Ghost positions were zeroed", closed_ghost_count == 22,
          f"{closed_ghost_count}/22 ghost hashes marked CLOSED_GHOST")
    check("No lingering ghost qty>0", open_count == 0,
          f"{open_count} hashes still have qty>0" if open_count > 0 else "all clean")
except Exception as e:
    check("Scan ghost positions", False, str(e))

# ─── 4. AUTONOMOUS TRADER: SLOT FIX ─────────────────────────
print("\n── 4. AUTONOMOUS TRADER — SLOT FIX ──")
try:
    result = subprocess.run(
        ['journalctl', '-u', 'quantum-autonomous-trader', '-n', '200', '--no-pager'],
        capture_output=True, text=True
    )
    logs = result.stdout

    # Get last SLOT_FIX report
    auth_count = None
    scanning_found = False
    entries_count = None
    
    for line in reversed(logs.splitlines()):
        if 'authoritative_count=' in line and auth_count is None:
            val = line.split('authoritative_count=')[1].split()[0]
            auth_count = int(val)
        if 'Scanning for entries' in line and not scanning_found:
            scanning_found = True
        if 'Stats:' in line and entries_count is None:
            parts = line.split('Stats:')[1]
            entries_count = parts.strip()
    
    print(f"   Last authoritative_count: {auth_count}")
    print(f"   Scanner active (slots>0): {scanning_found}")
    print(f"   Recent stats: {entries_count}")
    
    check("authoritative_count < 10 (not blocked)", 
          auth_count is not None and auth_count < 10,
          f"count={auth_count}" if auth_count is not None else "no data")
    check("Scanner is scanning entries (not blocked)", scanning_found,
          "Scanning for entries found in recent logs")
    
    # Get most recent entries/exits
    for line in reversed(logs.splitlines()):
        if 'entries,' in line and 'exits' in line:
            print(f"   Most recent: {line.strip()[-60:]}")
            break
except Exception as e:
    check("Read autonomous trader logs", False, str(e))

# ─── 5. NEW POSITIONS OPENED AFTER FIX ──────────────────────
print("\n── 5. NEW POSITIONS OPENED AFTER FIX ──")
try:
    result = subprocess.run(
        ['journalctl', '-u', 'quantum-autonomous-trader', '-n', '200', '--no-pager'],
        capture_output=True, text=True
    )
    logs = result.stdout
    new_positions = [l for l in logs.splitlines() if 'New position:' in l]
    print(f"   New positions in recent logs: {len(new_positions)}")
    for p in new_positions[-5:]:
        print(f"   {p.strip()[-70:]}")
    
    check("New positions opened after fix", len(new_positions) > 0,
          f"{len(new_positions)} new positions detected")
except Exception as e:
    check("Check new positions", False, str(e))

# ─── 6. SERVICES HEALTH ─────────────────────────────────────
print("\n── 6. KEY SERVICES ACTIVE ──")
services = [
    'quantum-intent-executor',
    'quantum-autonomous-trader',
    'quantum-apply-layer',
    'quantum-harvest-v2',
]
for svc in services:
    result = subprocess.run(['systemctl', 'is-active', svc], capture_output=True, text=True)
    active = result.stdout.strip() == 'active'
    check(f"{svc}", active, result.stdout.strip())

# ─── 7. PORTFOLIO STATE ──────────────────────────────────────
print("\n── 7. PORTFOLIO STATE ──")
portfolio = r.hgetall("quantum:state:portfolio")
if portfolio:
    bal = portfolio.get('balance_usd', '?')
    eq = portfolio.get('equity_usd', '?')
    pnl = portfolio.get('unrealized_pnl_usd', '?')
    pos = portfolio.get('positions_count', '?')
    print(f"   Balance:      ${bal}")
    print(f"   Equity:       ${eq}")
    print(f"   Unrealized:   ${pnl}")
    print(f"   Pos count:    {pos}")
    check("Portfolio state populated", bool(bal and bal != '?'), f"balance=${bal}")
else:
    check("Portfolio state populated", False, "quantum:state:portfolio is empty")

# ─── 8. HARVEST-V2 STATE ────────────────────────────────────
print("\n── 8. HARVEST-V2 (sandbox fix) ──")
try:
    result = subprocess.run(
        ['journalctl', '-u', 'quantum-harvest-v2', '-n', '20', '--no-pager'],
        capture_output=True, text=True
    )
    logs = result.stdout
    evaluated = [l for l in logs.splitlines() if 'evaluated=' in l]
    
    for line in evaluated[-2:]:
        print(f"   {line.strip()[-80:]}")
    
    check("harvest-v2 evaluating (not zero)", 
          any('evaluated=0' not in l for l in evaluated) and len(evaluated) > 0,
          f"{len(evaluated)} HV2_TICK lines found")
except Exception as e:
    check("harvest-v2 logs", False, str(e))

# ─── FINAL SUMMARY ──────────────────────────────────────────
print("\n" + "=" * 60)
passed = sum(1 for r, _, _ in results if r)
total = len(results)
print(f"RESULT: {passed}/{total} checks passed")

if passed == total:
    print("✅ ALL CHECKS PASSED — System is operational")
elif passed >= total * 0.8:
    print(f"⚠️  {total - passed} checks failed — minor issues")
else:
    print(f"❌ {total - passed} checks FAILED — investigate")
print("=" * 60)
