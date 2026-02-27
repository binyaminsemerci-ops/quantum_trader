#!/usr/bin/env python3
"""Fix indentation of post-close cooldown block (lines 2579-2582)"""
BASE = "/home/qt/quantum_trader/microservices/apply_layer/main.py"

with open(BASE, 'r', encoding='utf-8', errors='replace') as f:
    lines = f.readlines()

# Lines 2579-2582 (0-indexed: 2578-2581) have 38-space indent, need 32-space
# 38 spaces → 32 spaces
BAD_PREFIX  = ' ' * 38
GOOD_PREFIX = ' ' * 32

fixed = 0
for i in range(2577, 2584):  # lines around the injected block
    if lines[i].startswith(BAD_PREFIX):
        lines[i] = GOOD_PREFIX + lines[i][38:]
        print(f"  Fixed line {i+1}: {repr(lines[i][:60])}")
        fixed += 1

# Also fix double-braces in logger f-string (regex added {{ but f-string needs {)
for i in range(2577, 2584):
    if '{{symbol}}' in lines[i] or '{{_exit_cd}}' in lines[i]:
        lines[i] = lines[i].replace('{{symbol}}', '{symbol}').replace('{{_exit_cd}}', '{_exit_cd}')
        print(f"  Fixed braces line {i+1}: {repr(lines[i][:80])}")

print(f"\nTotal indentation fixes: {fixed}")

with open(BASE, 'w', encoding='utf-8') as f:
    f.writelines(lines)
print("File written.")

# Syntax check
import subprocess
r = subprocess.run(['python3', '-m', 'py_compile', BASE], capture_output=True, text=True)
if r.returncode == 0:
    print("✅ Syntax OK")
else:
    print(f"❌ Syntax error: {r.stderr}")
    # Restore backup
    import shutil
    shutil.copy(BASE + ".bak_cooldown", BASE)
    print("⚠️  Restored from backup")
    r2 = subprocess.run(['python3', '-m', 'py_compile', BASE], capture_output=True, text=True)
    print(f"Backup syntax: {'OK' if r2.returncode==0 else r2.stderr}")
