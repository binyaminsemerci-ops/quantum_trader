#!/usr/bin/env python3
"""
Force insert PARTIAL_25 step builder case before UPDATE_SL in the step builder.
Targets the unique pattern inside the 'if decision == Decision.EXECUTE:' block.
"""
import shutil, time

TARGET = "/opt/quantum/microservices/apply_layer/main.py"
shutil.copy2(TARGET, TARGET + f".bak3.{int(time.time())}")

with open(TARGET, "r") as f:
    content = f.read()

# The PARTIAL_25 step builder case to insert
PARTIAL25_BLOCK = '''
            elif action == "PARTIAL_25":
                steps.append({
                    "step": "CLOSE_PARTIAL_25",
                    "type": "market_reduce_only",
                    "side": "close",
                    "pct": 25.0
                })
'''

# Already patched?
if '"PARTIAL_25"' in content and 'step": "CLOSE_PARTIAL_25"' in content:
    print("[SKIP] PARTIAL_25 step builder already present")
    exit(0)

# Target the exact section: PARTIAL_50 block end + UPDATE_SL as anchor
# Pattern: the }) that closes PARTIAL_50's steps.append, followed by blank line, followed by elif action == "UPDATE_SL"
anchor = '                })\n\n            elif action == "UPDATE_SL":'
replacement = '                })\n' + PARTIAL25_BLOCK + '\n            elif action == "UPDATE_SL":'

if anchor in content:
    new_content = content.replace(anchor, replacement, 1)
    with open(TARGET, "w") as f:
        f.write(new_content)
    print("[OK] PATCH 5 FORCED: PARTIAL_25 step builder inserted before UPDATE_SL")
else:
    # Try alternative: single newline between }) and UPDATE_SL
    anchor2 = '                })\n            elif action == "UPDATE_SL":'
    replacement2 = '                })\n' + PARTIAL25_BLOCK + '            elif action == "UPDATE_SL":'
    if anchor2 in content:
        new_content = content.replace(anchor2, replacement2, 1)
        with open(TARGET, "w") as f:
            f.write(new_content)
        print("[OK] PATCH 5 FORCED (alt): PARTIAL_25 step builder inserted")
    else:
        # Last resort: find the exact location by searching for PARTIAL_50 pct line
        old = '                    "pct": 50.0\n                })\n'
        new = '                    "pct": 50.0\n                })\n\n            elif action == "PARTIAL_25":\n                steps.append({\n                    "step": "CLOSE_PARTIAL_25",\n                    "type": "market_reduce_only",\n                    "side": "close",\n                    "pct": 25.0\n                })\n'
        count = content.count(old)
        print(f"Last resort: found {count} occurrences of 50.0 pct block")
        if count == 1:
            new_content = content.replace(old, new, 1)
            with open(TARGET, "w") as f:
                f.write(new_content)
            print("[OK] PATCH 5 FORCED (last resort): inserted via pct:50.0 anchor")
        else:
            print(f"[ERROR] Cannot safely patch — {count} occurrences found")
            exit(1)

# Verify
with open(TARGET, "r") as f:
    final = f.read()
if 'CLOSE_PARTIAL_25' in final:
    print(f"[VERIFIED] CLOSE_PARTIAL_25 now in file")
else:
    print("[ERROR] CLOSE_PARTIAL_25 still not in file!")
    exit(1)
