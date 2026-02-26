#!/usr/bin/env python3
"""
Fix: Remove the bare 'n' on line 39 of exit_math.py that causes NameError.
This stray character was left from a prior session's ExitPosition alias edit.
"""
import py_compile
import sys

path = "/home/qt/quantum_trader/common/exit_math.py"

with open(path) as f:
    lines = f.readlines()

before = len(lines)
new_lines = [l for l in lines if l.strip() != "n"]
removed = before - len(new_lines)

if removed == 0:
    print("WARNING: No bare 'n' line found — may already be fixed or different content")
    # Show surrounding context
    for i, l in enumerate(lines[35:50], 36):
        print(f"  L{i}: {repr(l)}")
    sys.exit(1)

with open(path, "w") as f:
    f.writelines(new_lines)

print(f"REMOVED {removed} bare-n line(s) from exit_math.py")

# Syntax check
try:
    py_compile.compile(path, doraise=True)
    print("SYNTAX_OK")
except py_compile.PyCompileError as e:
    print(f"SYNTAX_ERROR: {e}")
    sys.exit(1)

# Show the fixed area
with open(path) as f:
    all_lines = f.readlines()

print("\nFixed context (lines 36-44):")
for i, l in enumerate(all_lines[35:44], 36):
    print(f"  L{i}: {l}", end="")
