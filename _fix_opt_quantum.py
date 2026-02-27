#!/usr/bin/env python3
"""
Apply FIX-A, FIX-B, FIX-C to /opt/quantum/microservices/harvest_brain/harvest_brain.py
Uses line-by-line approach to avoid multiline match fragility.
"""
import py_compile
import sys

path = "/opt/quantum/microservices/harvest_brain/harvest_brain.py"

with open(path) as f:
    lines = f.readlines()

fixes_applied = []
fix_a_done = fix_b_done = fix_c1_done = fix_c2_done = False

new_lines = []
i = 0
while i < len(lines):
    line = lines[i]

    # FIX-A: Insert R_net after trough_price in Position dataclass
    if (not fix_a_done and
            'trough_price: float = 0.0' in line and
            'Lowest price' in line):
        new_lines.append(line)
        next_chunk = ''.join(lines[i+1:i+5])
        if 'R_net: float = 0.0' not in next_chunk:
            new_lines.append('    R_net: float = 0.0          # Current R-multiple (set after P2 runs)\n')
            fixes_applied.append("FIX-A: R_net field added to Position dataclass")
        else:
            fixes_applied.append("FIX-A: already present")
        fix_a_done = True
        i += 1
        continue

    # FIX-B: Insert position.R_net = r_net after r_net = p2_result['R_net']
    if (not fix_b_done and
            "r_net = p2_result['R_net']" in line):
        new_lines.append(line)
        if i + 1 < len(lines) and 'position.R_net = r_net' not in lines[i+1]:
            new_lines.append("        position.R_net = r_net  # attach P2 R_net for _calculate_dynamic_fraction\n")
            fixes_applied.append("FIX-B: position.R_net = r_net setter added")
        else:
            fixes_applied.append("FIX-B: already present")
        fix_b_done = True
        i += 1
        continue

    # FIX-C1: insert pnl_per_unit before '# Build P2 PositionSnapshot'
    if (not fix_c1_done and
            '# Build P2 PositionSnapshot' in line):
        prev_chunk = ''.join(lines[max(0,i-3):i])
        if 'pnl_per_unit' not in prev_chunk:
            new_lines.append('        # FIX-C: P2 needs per-unit PnL, not total dollar PnL\n')
            new_lines.append('        pnl_per_unit = position.unrealized_pnl / max(position.qty, 1e-9)\n')
            fixes_applied.append("FIX-C1: pnl_per_unit line inserted")
        else:
            fixes_applied.append("FIX-C1: already present")
        fix_c1_done = True
        new_lines.append(line)
        i += 1
        continue

    # FIX-C2: replace unrealized_pnl=position.unrealized_pnl inside PositionSnapshot
    if (not fix_c2_done and fix_c1_done and
            'unrealized_pnl=position.unrealized_pnl' in line):
        new_lines.append(line.replace('unrealized_pnl=position.unrealized_pnl', 'unrealized_pnl=pnl_per_unit'))
        fixes_applied.append("FIX-C2: unrealized_pnl=pnl_per_unit in PositionSnapshot")
        fix_c2_done = True
        i += 1
        continue

    new_lines.append(line)
    i += 1

print("Fixes applied:")
for f in fixes_applied:
    print(" ", f)

if not (fix_a_done and fix_b_done and fix_c1_done and fix_c2_done):
    missing = []
    if not fix_a_done: missing.append("FIX-A (trough_price not found)")
    if not fix_b_done: missing.append("FIX-B (r_net = p2_result not found)")
    if not fix_c1_done: missing.append("FIX-C1 (PositionSnapshot comment not found)")
    if not fix_c2_done: missing.append("FIX-C2 (unrealized_pnl not found after C1)")
    print(f"\nFAILED to apply: {missing}")
    sys.exit(1)

with open(path, "w") as f:
    f.writelines(new_lines)

try:
    py_compile.compile(path, doraise=True)
    print("\nSYNTAX_OK: all fixes applied to /opt/quantum version")
except py_compile.PyCompileError as e:
    print(f"\nSYNTAX_ERROR: {e}")
    sys.exit(1)
