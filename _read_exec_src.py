#!/usr/bin/env python3
"""Read execution_service.py and show timestamp-related code + fix"""
import re

SRC = "/home/qt/quantum_trader/services/execution_service.py"

with open(SRC) as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}")
print("\n=== ALL lines mentioning timestamp/TTL/ISO/skip/SKIP ===")
for i, line in enumerate(lines, 1):
    lower = line.lower()
    if any(k in lower for k in ["timestamp", "ttl", "iso time", "too short", "decision", "skip", "would_execute", "parse"]):
        print(f"  {i:4}: {line.rstrip()}")

print("\n=== Context around PATH1B (lines -5 to +20 of first RX) ===")
for i, line in enumerate(lines, 1):
    if "PATH1B" in line or "RX apply" in line:
        start = max(0, i-10)
        end = min(len(lines), i+30)
        print(f"\n--- Found PATH1B at line {i} ---")
        for j in range(start, end):
            print(f"  {j+1:4}: {lines[j].rstrip()}")
        break
