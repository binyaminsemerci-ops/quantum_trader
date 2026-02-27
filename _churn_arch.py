#!/usr/bin/env python3
"""
Dybdediagnose av churning arkitektur:
1. Autonomous trader exit_manager — hva setter exit threshold
2. Apply layer open cooldown — 180s hardkodet?
3. AI Engine timeout fallback — konfigurbar?
"""
import os, redis

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

def show_file_section(path, terms, context=3):
    try:
        with open(path) as f:
            lines = f.readlines()
    except:
        print(f"  CANNOT READ: {path}")
        return
    for i, line in enumerate(lines):
        if any(t in line for t in terms) and not line.strip().startswith("#"):
            start = max(0, i - context)
            end = min(len(lines), i + context + 1)
            print(f"\n  [{path.split('quantum_trader/')[-1]}:{i+1}]")
            for j in range(start, end):
                prefix = ">>>" if j == i else "   "
                print(f"  {prefix} {lines[j].rstrip()[:100]}")

print("=" * 65)
print("CHURNING ARKITEKTUR DIAGNOSE")
print("=" * 65)

# ─── 1. AUTONOMOUS TRADER: EXIT MANAGER ──────────────────────
print("\n── 1. EXIT MANAGER: R-THRESHOLD FALLBACK ──")
at_dir = "/home/qt/quantum_trader/microservices/autonomous_trader/"
exit_mgr = at_dir + "exit_manager.py"

# Finn fallback logikk
show_file_section(exit_mgr,
    ["fallback", "R_threshold", "r_threshold", "exit_threshold", "CLOSE",
     "timeout", "AI_ENGINE", "ai_engine_timeout", "hold_count", "hold_bars"],
    context=4)

print("\n── 2. AUTONOMOUS TRADER: HOLD CONFIG ──")
main_file = at_dir + "autonomous_trader.py"
show_file_section(main_file,
    ["hold_count", "min_hold", "MIN_HOLD", "hold_bars", "HOLD_THRESHOLD",
     "exit_score_threshold", "exit_threshold", "R_threshold"],
    context=3)

# ─── 2. APPLY LAYER: OPEN COOLDOWN TTL ───────────────────────
print("\n── 3. APPLY LAYER: OPEN COOLDOWN ──")
apply_file = "/home/qt/quantum_trader/microservices/apply_layer/main.py"
show_file_section(apply_file,
    ["cooldown:open", "setex", "OPEN_COOLDOWN", "cooldown_sec",
     "COOLDOWN_SEC", "ENTRY_COOLDOWN", "recently_opened"],
    context=4)

# ─── 3. APPLY LAYER: CLOSE COOLDOWN ──────────────────────────
print("\n── 4. APPLY LAYER: CLOSE COOLDOWN (finnes?) ──")
show_file_section(apply_file,
    ["cooldown:close", "cooldown:last", "after_close", "post_close",
     "CLOSE_COOLDOWN", "reentry", "re_entry", "RE_ENTRY"],
    context=4)

# ─── 4. INTENT EXECUTOR: COOLDOWN ────────────────────────────
print("\n── 5. INTENT EXECUTOR: COOLDOWN LOGIKK ──")
ie_file = "/home/qt/quantum_trader/microservices/intent_executor/main.py"
show_file_section(ie_file,
    ["cooldown", "last_exec_ts", "COOLDOWN", "MIN_INTERVAL", "min_interval"],
    context=5)

# ─── 5. ENV KONFIGURASJON ────────────────────────────────────
print("\n── 6. KONFIGURASJON: ENV VARS ──")
print("\n  apply-layer.env (cooldown/hold relatert):")
with open("/etc/quantum/apply-layer.env") as f:
    for line in f:
        line = line.strip()
        if any(t in line.upper() for t in ["COOLDOWN", "HOLD", "INTERVAL", "TTL", "MIN_", "REENTRY", "LOCK"]):
            if not line.startswith("#") and "=" in line:
                print(f"  {line}")

print("\n  autonomous-trader.env (hvis finnes):")
for env_path in ["/etc/quantum/autonomous-trader.env", "/etc/quantum/autonomous_trader.env"]:
    if os.path.exists(env_path):
        with open(env_path) as f:
            print(f"  {env_path}:")
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    print(f"  {line}")
    else:
        print(f"  {env_path}: IKKE FUNNET")

# ─── 6. REDIS KONFIG ─────────────────────────────────────────
print("\n── 7. REDIS: AKTIVE COOLDOWN NØKLER ──")
cooldown_keys = r.keys("quantum:cooldown:*")
print(f"  Aktive cooldown keys: {len(cooldown_keys)}")
for k in sorted(cooldown_keys)[:15]:
    ttl = r.ttl(k)
    val = r.get(k) or r.hgetall(k)
    print(f"  {k}  TTL={ttl}s  val={str(val)[:30]}")

# ─── 7. OPEN COOLDOWN TTL VERDI ──────────────────────────────
print("\n── 8. OPEN COOLDOWN TTL I KODE ──")
import subprocess
result = subprocess.run(
    ["grep", "-n", "setex.*cooldown", apply_file],
    capture_output=True, text=True
)
print(result.stdout or "  Ingen setex cooldown funnet")

result2 = subprocess.run(
    ["grep", "-n", "ENTRY_COOLDOWN\|OPEN_COOLDOWN\|cooldown.*sec\|getenv.*cooldown",
     apply_file],
    capture_output=True, text=True
)
print(result2.stdout or "  Ingen env cooldown konfig funnet")
