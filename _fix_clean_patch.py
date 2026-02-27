#!/usr/bin/env python3
"""
Clean indentation-aware patch for apply_layer/main.py.
Reads actual surrounding indent and matches it.
"""
import subprocess, sys

BASE = "/home/qt/quantum_trader/microservices/apply_layer/main.py"
BACKUP = BASE + ".bak_cooldown"

with open(BASE, 'r', encoding='utf-8', errors='replace') as f:
    lines = f.readlines()

print(f"Lines total: {len(lines)}")

changes = []

# ── PATCH 1: Entry cooldown 180 → env var ──────────────────────
# Find the line with: self.redis.setex(cooldown_key, 180, "1")
for i, line in enumerate(lines):
    stripped = line.strip()
    if stripped == 'self.redis.setex(cooldown_key, 180, "1")':
        indent = line[:len(line) - len(line.lstrip())]
        # Replace this ONE line with TWO lines at the same indent
        new_lines = [
            f'{indent}_entry_cd = int(__import__("os").getenv("APPLY_ENTRY_COOLDOWN_SEC", "600"))\n',
            f'{indent}self.redis.setex(cooldown_key, _entry_cd, "1")\n',
        ]
        # Also find the logger line right after and update it
        lines[i] = new_lines[0] + new_lines[1]
        changes.append(f"Patch1 @ line {i+1}: entry cooldown 180→env")
        print(f"[P1] Line {i+1}: {repr(indent)}  ({len(indent)} spaces)")
        # Update the logger line if it's the next non-empty line
        for j in range(i+1, min(i+4, len(lines))):
            if 'Cooldown set (180s)' in lines[j]:
                lines[j] = lines[j].replace('Cooldown set (180s)', 'Cooldown set ({_entry_cd}s)')
                changes.append(f"Patch1b @ line {j+1}: logger updated")
                print(f"[P1b] Line {j+1}: logger updated")
                break
        break

# ── PATCH 2: Post-close entry cooldown ─────────────────────────
# Find: "# Set dedupe marker" followed by setex(dedupe_key, 600, "1")
# in the CLOSE SUCCESS block (near "Position updated" log)
# Strategy: find all "# Set dedupe marker" + "setex(dedupe_key, 600" combos
# and inject cooldown after the one that follows "Position updated"
target_context_found = False
for i, line in enumerate(lines):
    if 'logger.info' in line and 'Position updated' in line and 'qty=' in line:
        # Look forward for the dedupe setex
        for j in range(i+1, min(i+20, len(lines))):
            stripped_j = lines[j].strip()
            if stripped_j.startswith('self.redis.setex(dedupe_key'):
                indent = lines[j][:len(lines[j]) - len(lines[j].lstrip())]
                # Insert after this line
                cooldown_lines = (
                    f'{indent}# Post-close re-entry cooldown (prevents churn)\n'
                    f'{indent}_exit_cd = int(__import__("os").getenv("APPLY_EXIT_COOLDOWN_SEC", "600"))\n'
                    f'{indent}self.redis.setex(f"quantum:cooldown:open:{{symbol}}", _exit_cd, "1")\n'
                    f'{indent}logger.info(f"[CLOSE] {{symbol}}: Post-exit cooldown ({{_exit_cd}}s)")\n'
                )
                lines[j] = lines[j] + cooldown_lines
                changes.append(f"Patch2 @ line {j+1}: post-close cooldown injected")
                print(f"[P2] Line {j+1}: post-close cooldown injected ({len(indent)} spaces indent)")
                target_context_found = True
                break
        if target_context_found:
            break

if not target_context_found:
    print("[P2] WARNING: Could not find 'Position updated qty=' context — skipping post-close cooldown")

# ── Write ──────────────────────────────────────────────────────
with open(BASE, 'w', encoding='utf-8') as f:
    f.writelines(lines)
print(f"\nWrote {len(lines)} lines")

# ── Syntax check ───────────────────────────────────────────────
import shutil
r = subprocess.run(['python3', '-m', 'py_compile', BASE], capture_output=True, text=True)
if r.returncode == 0:
    print("✅ Syntax OK")
    print(f"\nChanges applied:")
    for c in changes:
        print(f"  {c}")
else:
    print(f"❌ Syntax error: {r.stderr}")
    shutil.copy(BACKUP, BASE)
    print("⚠️  Restored from backup")
    r2 = subprocess.run(['python3', '-m', 'py_compile', BASE], capture_output=True, text=True)
    print(f"Backup syntax: {'OK' if r2.returncode==0 else r2.stderr}")
    sys.exit(1)
