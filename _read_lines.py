#!/usr/bin/env python3
"""
Fix indentation error in apply_layer/main.py at line ~2580
and restore from backup if needed, then apply clean patch.
"""
import os, sys, re, subprocess

BASE = "/home/qt/quantum_trader/microservices/apply_layer/main.py"
BACKUP = BASE + ".bak_cooldown"

# Read current (broken) file
with open(BASE, 'r', encoding='utf-8', errors='replace') as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}")
print(f"Lines 2568-2595:")
for i, line in enumerate(lines[2568:2595], start=2569):
    print(f"  {i:4d}|{repr(line)}")
