#!/usr/bin/env python3
"""
Patch apply_layer/main.py:
1. Add PARTIAL_25 to normalize_action pass-through (line ~1197)
2. Add PARTIAL_25 to is_close_action check (line ~1266)
3. Add PARTIAL_25 to Gate 5 close exclusion list (line ~1315)
4. Add PARTIAL_25 to Gate 6 close exclusion list (line ~1335)
5. Add PARTIAL_25 step builder case after PARTIAL_50 block (line ~1388)
"""
import re, sys, shutil, time

TARGET = "/opt/quantum/microservices/apply_layer/main.py"

# Backup
shutil.copy2(TARGET, TARGET + f".bak.{int(time.time())}")
print(f"[OK] Backup created")

with open(TARGET, "r") as f:
    src = f.read()

changes = 0

# ── PATCH 1: normalize_action pass-through list ──────────────────────────────
OLD1 = 'elif action in ["FULL_CLOSE_PROPOSED", "PARTIAL_75", "PARTIAL_50", "UPDATE_SL", "HOLD"]:'
NEW1 = 'elif action in ["FULL_CLOSE_PROPOSED", "PARTIAL_75", "PARTIAL_50", "PARTIAL_25", "UPDATE_SL", "HOLD"]:'
if OLD1 in src:
    src = src.replace(OLD1, NEW1, 1)
    print("[OK] PATCH 1: normalize_action pass-through — PARTIAL_25 added")
    changes += 1
else:
    print("[WARN] PATCH 1: target string not found — skipping")

# ── PATCH 2: is_close_action check ──────────────────────────────────────────
OLD2 = 'is_close_action = action in ["FULL_CLOSE_PROPOSED", "PARTIAL_75", "PARTIAL_50"]'
NEW2 = 'is_close_action = action in ["FULL_CLOSE_PROPOSED", "PARTIAL_75", "PARTIAL_50", "PARTIAL_25"]'
if OLD2 in src:
    src = src.replace(OLD2, NEW2, 1)
    print("[OK] PATCH 2: is_close_action — PARTIAL_25 added")
    changes += 1
else:
    print("[WARN] PATCH 2: target string not found — skipping")

# ── PATCH 3 & 4: Gate 5+6 close exclusion tuples ────────────────────────────
OLD_GATE = "                'FULL_CLOSE_PROPOSED', 'PARTIAL_75', 'PARTIAL_50', 'UPDATE_SL', 'HOLD'\n"
NEW_GATE = "                'FULL_CLOSE_PROPOSED', 'PARTIAL_75', 'PARTIAL_50', 'PARTIAL_25', 'UPDATE_SL', 'HOLD'\n"
count_gate = src.count(OLD_GATE)
if count_gate >= 1:
    src = src.replace(OLD_GATE, NEW_GATE)  # replaces all occurrences (gate 5 + gate 6)
    print(f"[OK] PATCH 3+4: Gate 5+6 close exclusion — PARTIAL_25 added ({count_gate} occurrence(s))")
    changes += count_gate
else:
    print("[WARN] PATCH 3+4: gate exclusion tuple not found — trying alternative format")
    alt = "                'FULL_CLOSE_PROPOSED', 'PARTIAL_75', 'PARTIAL_50', 'UPDATE_SL', 'HOLD'"
    alt_new = "                'FULL_CLOSE_PROPOSED', 'PARTIAL_75', 'PARTIAL_50', 'PARTIAL_25', 'UPDATE_SL', 'HOLD'"
    count_alt = src.count(alt)
    if count_alt >= 1:
        src = src.replace(alt, alt_new)
        print(f"[OK] PATCH 3+4 (alt): PARTIAL_25 added ({count_alt} occurrence(s))")
        changes += count_alt
    else:
        print("[WARN] PATCH 3+4: nothing patched — manual inspection needed")

# ── PATCH 5: Step builder — add PARTIAL_25 case after PARTIAL_50 ─────────────
OLD5 = '''            elif action == "PARTIAL_50":
                steps.append({
                    "step": "CLOSE_PARTIAL_50",
                    "type": "market_reduce_only",
                    "side": "close",
                    "pct": 50.0
                })

            elif action == "UPDATE_SL":'''
NEW5 = '''            elif action == "PARTIAL_50":
                steps.append({
                    "step": "CLOSE_PARTIAL_50",
                    "type": "market_reduce_only",
                    "side": "close",
                    "pct": 50.0
                })

            elif action == "PARTIAL_25":
                steps.append({
                    "step": "CLOSE_PARTIAL_25",
                    "type": "market_reduce_only",
                    "side": "close",
                    "pct": 25.0
                })

            elif action == "UPDATE_SL":'''
if OLD5 in src:
    src = src.replace(OLD5, NEW5, 1)
    print("[OK] PATCH 5: Step builder PARTIAL_25 case added")
    changes += 1
else:
    print("[WARN] PATCH 5: PARTIAL_50 step block not found — skipping")

if changes > 0:
    with open(TARGET, "w") as f:
        f.write(src)
    print(f"\n[DONE] {changes} patch(es) applied to {TARGET}")
else:
    print("\n[ERROR] No patches applied!")
    sys.exit(1)
