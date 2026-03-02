#!/usr/bin/env python3
"""
Patch 5: Insert PARTIAL_25 step builder case after PARTIAL_50 block (line 1393)
Uses line number targeting for precision.
"""
import shutil, time

TARGET = "/opt/quantum/microservices/apply_layer/main.py"
shutil.copy2(TARGET, TARGET + f".bak2.{int(time.time())}")

with open(TARGET, "r") as f:
    lines = f.readlines()

# Verify already has PARTIAL_25 step builder?
already_patched = any('"PARTIAL_25"' in l and 'action ==' in l for l in lines)
if already_patched:
    print("[SKIP] PARTIAL_25 step builder already present")
    exit(0)

# Find the PARTIAL_50 block end (line 1393 = index 1392)
insert_after = None
for i, line in enumerate(lines):
    if '"PARTIAL_50"' in line and 'action ==' in line:
        # Look for the closing }) and blank line
        for j in range(i, min(len(lines), i+10)):
            if lines[j].strip() == '})':
                insert_after = j  # insert after this line
                break
        break

if insert_after is None:
    print("[ERROR] Could not find PARTIAL_50 block end")
    exit(1)

print(f"[OK] Found PARTIAL_50 block end at line {insert_after+1}")
print(f"[OK] Inserting PARTIAL_25 block after line {insert_after+1}")

# Build PARTIAL_25 block (same indentation as PARTIAL_50)
partial25_block = [
    '            \n',
    '            elif action == "PARTIAL_25":\n',
    '                steps.append({\n',
    '                    "step": "CLOSE_PARTIAL_25",\n',
    '                    "type": "market_reduce_only",\n',
    '                    "side": "close",\n',
    '                    "pct": 25.0\n',
    '                })\n',
]

# Insert after the }) line (insert_after index)
new_lines = lines[:insert_after+1] + partial25_block + lines[insert_after+1:]

with open(TARGET, "w") as f:
    f.writelines(new_lines)

# Verify
with open(TARGET, "r") as f:
    content = f.read()
count = content.count('"PARTIAL_25"')
print(f"[OK] PATCH 5 applied — 'PARTIAL_25' now appears {count} time(s) in file")
