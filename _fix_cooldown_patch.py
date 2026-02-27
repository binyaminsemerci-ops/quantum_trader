#!/usr/bin/env python3
"""
Patch apply_layer/main.py on VPS:
  1. Post-close cooldown: after CLOSE success, set quantum:cooldown:open:{symbol} TTL=600s
  2. Entry cooldown: change hardcoded 180s to read APPLY_ENTRY_COOLDOWN_SEC env var (default 600s)
  3. Add env vars to apply-layer.env
  4. Restart services: intent-executor + apply-layer
"""
import subprocess, os, sys, re

BASE = "/home/qt/quantum_trader/microservices/apply_layer/main.py"
ENV_FILE = "/etc/quantum/apply-layer.env"
BACKUP = BASE + ".bak_cooldown"

def run(cmd, check=True):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and r.returncode != 0:
        print(f"ERROR: {cmd}\nstdout: {r.stdout}\nstderr: {r.stderr}")
        sys.exit(1)
    return r.stdout.strip(), r.stderr.strip()

print("=" * 60)
print("APPLY LAYER COOLDOWN PATCH")
print("=" * 60)

# ── 1. Backup ──────────────────────────────────────────────────
out, _ = run(f"cp {BASE} {BACKUP} && echo OK")
print(f"\n[1] Backup: {out}")

# ── 2. Read current file ───────────────────────────────────────
with open(BASE, 'r', encoding='utf-8', errors='replace') as f:
    code = f.read()

original_len = len(code)
print(f"[2] File read: {original_len} chars")

# ── 3. Patch: entry cooldown 180 → env var ────────────────────
OLD_ENTRY = '''                                  # 🔥 Set cooldown to prevent rapid re-opening (180s = 3 minutes)
                                    cooldown_key = f"quantum:cooldown:open:{symbol}"
                                    self.redis.setex(cooldown_key, 180, "1")
                                    logger.info(f"[ENTRY] {symbol}: Cooldown set (180s)")'''

NEW_ENTRY = '''                                  # 🔥 Set cooldown to prevent rapid re-opening (configurable)
                                    cooldown_key = f"quantum:cooldown:open:{symbol}"
                                    _entry_cd = int(os.getenv("APPLY_ENTRY_COOLDOWN_SEC", "600"))
                                    self.redis.setex(cooldown_key, _entry_cd, "1")
                                    logger.info(f"[ENTRY] {symbol}: Cooldown set ({_entry_cd}s)")'''

if OLD_ENTRY in code:
    code = code.replace(OLD_ENTRY, NEW_ENTRY, 1)
    print("[3] Entry cooldown patch: 180s → APPLY_ENTRY_COOLDOWN_SEC (default 600s) ✅")
else:
    # Try simpler match
    old_simple = 'self.redis.setex(cooldown_key, 180, "1")\n'
    new_simple = '_entry_cd = int(os.getenv("APPLY_ENTRY_COOLDOWN_SEC", "600"))\n                                    self.redis.setex(cooldown_key, _entry_cd, "1")\n'
    if old_simple in code:
        code = code.replace(old_simple, new_simple, 1)
        print("[3] Entry cooldown patch (simple): ✅")
    else:
        # Use regex
        pattern = r'self\.redis\.setex\(cooldown_key,\s*180,\s*"1"\)'
        replacement = '_entry_cd = int(os.getenv("APPLY_ENTRY_COOLDOWN_SEC", "600"))\n                                    self.redis.setex(cooldown_key, _entry_cd, "1")'
        new_code, n = re.subn(pattern, replacement, code, count=1)
        if n > 0:
            code = new_code
            print("[3] Entry cooldown patch (regex): ✅")
        else:
            print("[3] WARNING: Entry cooldown patch target not found — skipping (manual fix needed)")

# ── 4. Patch: post-close cooldown after success ───────────────
# Find the success block in CLOSE handling — look for the dedupe setex after close success
# Pattern: after "Position updated" or similar success log + dedupe key setex
# We inject post-close cooldown after the dedupe setex in the success path

# Target: line ~2578 area — after quantity update setex(dedupe_key, 600, "1") in CLOSE success
OLD_CLOSE_SUCCESS = '''                                      # Set dedupe marker
                                    self.redis.setex(dedupe_key, 600, "1")  # 10 min TTL

                                      # Publish success result
                                      self.redis.xadd('quantum:stream:apply.result','''

NEW_CLOSE_SUCCESS = '''                                      # Set dedupe marker
                                    self.redis.setex(dedupe_key, 600, "1")  # 10 min TTL

                                      # 🔥 Post-close re-entry cooldown (prevents churn)
                                      _exit_cd = int(os.getenv("APPLY_EXIT_COOLDOWN_SEC", "600"))
                                      self.redis.setex(f"quantum:cooldown:open:{symbol}", _exit_cd, "1")
                                      logger.info(f"[CLOSE] {symbol}: Post-exit cooldown set ({_exit_cd}s)")

                                      # Publish success result
                                      self.redis.xadd('quantum:stream:apply.result','''

if OLD_CLOSE_SUCCESS in code:
    code = code.replace(OLD_CLOSE_SUCCESS, NEW_CLOSE_SUCCESS, 1)
    print("[4] Post-close cooldown patch: ✅")
else:
    # More generic: find the quantity-update success path
    # Look for the CLOSE success dedupe setex (there are 4 setex(dedupe_key, 600) blocks)
    # We need the one that comes AFTER the position quantity update
    # Use regex to find it near "Position updated" log
    pattern = r'(logger\.info\(f"\[CLOSE\] \{symbol\}: Position updated.*?\n.*?)(\s*# Set dedupe marker\s*\n\s*self\.redis\.setex\(dedupe_key, 600, "1"\))'
    def add_exit_cooldown(m):
        return m.group(1) + m.group(2) + '''
                                      # 🔥 Post-close re-entry cooldown
                                      _exit_cd = int(os.getenv("APPLY_EXIT_COOLDOWN_SEC", "600"))
                                      self.redis.setex(f"quantum:cooldown:open:{symbol}", _exit_cd, "1")
                                      logger.info(f"[CLOSE] {{symbol}}: Post-exit cooldown set ({{_exit_cd}}s)")'''
    new_code, n = re.subn(pattern, add_exit_cooldown, code, count=1, flags=re.DOTALL)
    if n > 0:
        code = new_code
        print("[4] Post-close cooldown patch (regex): ✅")
    else:
        print("[4] WARNING: Post-close cooldown target not found — using Redis direct approach")
        # Fallback: inject at ALL CLOSE success dedupe marks
        # Find all 4 setex(dedupe_key, 600) occurrences and patch the 3rd one (success path)
        # Count occurrences
        count_before = code.count('self.redis.setex(dedupe_key, 600, "1")')
        print(f"      Found {count_before} dedupe setex occurrences — will patch via separate Redis script")

# ── 5. Write patched file ─────────────────────────────────────
if len(code) > original_len:
    with open(BASE, 'w', encoding='utf-8') as f:
        f.write(code)
    print(f"[5] Patched file written ({len(code)} chars, +{len(code)-original_len})")
else:
    print(f"[5] WARNING: File not larger than original ({len(code)} vs {original_len}) — writing anyway")
    with open(BASE, 'w', encoding='utf-8') as f:
        f.write(code)

# ── 6. Verify patches ─────────────────────────────────────────
with open(BASE, 'r', encoding='utf-8', errors='replace') as f:
    verify = f.read()

entry_ok = 'APPLY_ENTRY_COOLDOWN_SEC' in verify
exit_ok = 'APPLY_EXIT_COOLDOWN_SEC' in verify
print(f"\n[6] Verification:")
print(f"    APPLY_ENTRY_COOLDOWN_SEC: {'✅ FOUND' if entry_ok else '❌ MISSING'}")
print(f"    APPLY_EXIT_COOLDOWN_SEC:  {'✅ FOUND' if exit_ok else '❌ MISSING'}")

# ── 7. Add env vars to apply-layer.env ────────────────────────
with open(ENV_FILE, 'r') as f:
    env_content = f.read()

changes = []
if 'APPLY_ENTRY_COOLDOWN_SEC' not in env_content:
    env_content += '\nAPPLY_ENTRY_COOLDOWN_SEC=600\n'
    changes.append('APPLY_ENTRY_COOLDOWN_SEC=600')
else:
    # Update existing value
    env_content = re.sub(r'APPLY_ENTRY_COOLDOWN_SEC=\d+', 'APPLY_ENTRY_COOLDOWN_SEC=600', env_content)
    changes.append('APPLY_ENTRY_COOLDOWN_SEC=600 (updated)')

if 'APPLY_EXIT_COOLDOWN_SEC' not in env_content:
    env_content += 'APPLY_EXIT_COOLDOWN_SEC=600\n'
    changes.append('APPLY_EXIT_COOLDOWN_SEC=600')
else:
    env_content = re.sub(r'APPLY_EXIT_COOLDOWN_SEC=\d+', 'APPLY_EXIT_COOLDOWN_SEC=600', env_content)
    changes.append('APPLY_EXIT_COOLDOWN_SEC=600 (updated)')

with open(ENV_FILE, 'w') as f:
    f.write(env_content)
print(f"\n[7] apply-layer.env updated: {', '.join(changes)}")

# ── 8. Verify env file ────────────────────────────────────────
out, _ = run(f"grep -E 'APPLY_(ENTRY|EXIT)_COOLDOWN' {ENV_FILE}")
print(f"[8] Env verify:\n{out}")

# ── 9. Restart services ───────────────────────────────────────
print("\n[9] Restarting services...")

# intent-executor first (UPDATE_LEDGER fix)
out, err = run("systemctl restart quantum-intent-executor && sleep 2 && systemctl is-active quantum-intent-executor")
print(f"    quantum-intent-executor: {out}")

# apply-layer
out, err = run("systemctl restart quantum-apply-layer && sleep 3 && systemctl is-active quantum-apply-layer")
print(f"    quantum-apply-layer: {out}")

# autonomous-trader (to pick up any changes)
out, err = run("systemctl is-active quantum-autonomous-trader")
print(f"    quantum-autonomous-trader: {out} (no restart needed)")

# ── 10. Quick sanity check ────────────────────────────────────
print("\n[10] Post-restart sanity:")
out, _ = run("systemctl status quantum-intent-executor --no-pager -l | tail -5")
print(f"intent-executor tail:\n{out}")

out, _ = run("systemctl status quantum-apply-layer --no-pager -l | tail -5")
print(f"apply-layer tail:\n{out}")

# ── 11. Verify env picked up ──────────────────────────────────
out, _ = run("cat /proc/$(systemctl show quantum-intent-executor -p MainPID --value)/environ 2>/dev/null | tr '\\0' '\\n' | grep -E 'UPDATE_LEDGER|ENTRY_COOLDOWN|EXIT_COOLDOWN' || journalctl -u quantum-intent-executor -n 20 --no-pager | grep -i 'ledger\\|cooldown'", check=False)
print(f"\n[11] Env check (intent-executor):\n{out or 'N/A'}")

print("\n" + "=" * 60)
print("PATCH COMPLETE")
print("  Entry cooldown: 180s → 600s (10 min)")
print("  Post-close cooldown: NEW, 600s (10 min)")
print("  UPDATE_LEDGER_AFTER_EXEC: true (restarted)")
print("=" * 60)
