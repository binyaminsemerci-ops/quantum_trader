#!/usr/bin/env python3
"""
Fix two problems:
1. AI Engine down → restart it
2. Ghost slots re-accumulated → zero them
3. Show scanner config (why no entries)
"""
import redis, subprocess, time, os

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

print("=" * 60)
print("SYSTEM FIX: AI ENGINE + GHOST SLOTS")
print("=" * 60)

# ── 1. AI Engine health ────────────────────────────────────────
print("\n[1] AI ENGINE STATUS")
import urllib.request, json
try:
    with urllib.request.urlopen("http://127.0.0.1:8001/health", timeout=5) as resp:
        data = json.loads(resp.read())
        print(f"  Status: {data}")
        ai_ok = True
except Exception as e:
    print(f"  ❌ DOWN: {e}")
    ai_ok = False

# Find service name
for name in ["quantum-ai-engine", "ai-engine", "aiengine", "quantum_ai_engine"]:
    r2 = subprocess.run(["systemctl", "is-active", name], capture_output=True, text=True)
    if r2.stdout.strip() in ("active", "inactive", "failed", "activating"):
        ai_svc = name
        print(f"  Service: {name} → {r2.stdout.strip()}")
        break
else:
    # Search
    r2 = subprocess.run(["systemctl", "list-units", "--type=service", "--no-pager"],
                       capture_output=True, text=True)
    ai_svcs = [l for l in r2.stdout.splitlines() if "ai" in l.lower() and "engine" in l.lower()]
    print(f"  Found services: {ai_svcs[:3]}")
    ai_svc = None

if not ai_ok and ai_svc:
    print(f"  Restarting {ai_svc}...")
    subprocess.run(["systemctl", "restart", ai_svc])
    time.sleep(5)
    try:
        with urllib.request.urlopen("http://127.0.0.1:8001/health", timeout=8) as resp:
            data = json.loads(resp.read())
            print(f"  ✅ AI Engine now: {data}")
    except Exception as e:
        print(f"  Still down: {e}")
        # Check logs
        r2 = subprocess.run(["journalctl", "-u", ai_svc, "-n", "20", "--no-pager"],
                           capture_output=True, text=True)
        print(f"  Logs:\n{r2.stdout[-2000:]}")
elif not ai_ok and not ai_svc:
    print("  Could not find AI Engine service — checking processes")
    r2 = subprocess.run(["pgrep", "-fa", "ai_engine"], capture_output=True, text=True)
    print(f"  Process: {r2.stdout.strip() or 'NONE'}")
    # Try to start directly
    r2 = subprocess.run(["find", "/home/qt/quantum_trader/microservices", "-name", "main.py", "-path", "*/ai_engine/*"],
                       capture_output=True, text=True)
    print(f"  ai_engine main.py: {r2.stdout.strip()}")

# ── 2. Ghost slots ─────────────────────────────────────────────
print("\n[2] GHOST SLOTS CHECK")
pos_keys = r.keys("quantum:position:*")
ledger_key = "quantum:ledger:positions"

# Authoritative count from ledger
ledger_syms = r.smembers(ledger_key) if r.exists(ledger_key) else set()
print(f"  Ledger (authoritative): {len(ledger_syms)} → {sorted(ledger_syms)[:5]}")

# Legacy hashes
ghost_count = 0
ghosts = []
for k in pos_keys:
    try:
        d = r.hgetall(k)
        qty = float(d.get('quantity', 0))
        status = d.get('status', '')
        sym = k.split(':')[-1]
        if qty > 0 and status not in ('CLOSED', 'CLOSED_GHOST'):
            # Check if in ledger
            if sym not in ledger_syms:
                ghosts.append((sym, qty, status))
                ghost_count += 1
    except:
        pass

print(f"  Legacy qty>0 NOT in ledger: {ghost_count}")
if ghosts:
    print(f"  Ghosts to zero: {[g[0] for g in ghosts]}")
    for sym, qty, status in ghosts:
        k = f"quantum:position:{sym}"
        r.hset(k, mapping={'quantity': '0', 'status': 'CLOSED_GHOST'})
        print(f"    Zeroed: {sym} (was qty={qty}, status={status})")
    print(f"  ✅ {ghost_count} ghost slots cleared")
else:
    print("  ✅ No ghosts")

# ── 3. Slot count after fix ────────────────────────────────────
print("\n[3] SLOT COUNT AFTER FIX")
pos_keys2 = r.keys("quantum:position:*")
real_active = 0
for k in pos_keys2:
    try:
        d = r.hgetall(k)
        qty = float(d.get('quantity', 0))
        status = d.get('status', '')
        if qty > 0 and status not in ('CLOSED', 'CLOSED_GHOST'):
            sym = k.split(':')[-1]
            print(f"  ACTIVE: {sym} qty={qty} status={status}")
            real_active += 1
    except:
        pass
print(f"  Active positions: {real_active} / 10 max → {10-real_active} free slots")

# ── 4. Scanner config ──────────────────────────────────────────
print("\n[4] SCANNER CONFIG")
print(f"  MIN_CONFIDENCE from env: {os.getenv('MIN_CONFIDENCE', 'NOT SET (default 0.65)')}")
print(f"  USE_UNIVERSE_OS: {os.getenv('USE_UNIVERSE_OS', 'NOT SET')}")
print(f"  UNIVERSE_MAX_SYMBOLS: {os.getenv('UNIVERSE_MAX_SYMBOLS', 'NOT SET')}")

# Check universe
uni_key = r.keys("quantum:universe:*")
print(f"  Universe Redis keys: {uni_key[:5]}")

# Check P3.5 guard config
p35 = r.hgetall("quantum:config:p35") or {}
print(f"  P3.5 config: {p35}")

# Recent scan results
print("\n[5] RECENT SCANNER LOG CHECK")
r2 = subprocess.run(
    ["journalctl", "-u", "quantum-autonomous-trader", "-n", "100", "--no-pager",
     "--grep", "Scanner|SKIP|No entry|opportunity|confidence"],
    capture_output=True, text=True
)
scanner_lines = [l for l in r2.stdout.splitlines() if any(w in l for w in ['Scanner', 'SKIP', 'No entry', 'confidence', 'opportunity', 'MIN_CONF'])]
for l in scanner_lines[-10:]:
    # Extract just the message part
    parts = l.split('] ')
    print(f"  {parts[-1] if len(parts) > 1 else l}")

print("\n" + "=" * 60)
