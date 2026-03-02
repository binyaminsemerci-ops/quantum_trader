#!/usr/bin/env python3
"""
Read and print the exact text around the PARTIAL_50 step builder block
in the already-patched apply_layer/main.py to find the correct string
"""
TARGET = "/opt/quantum/microservices/apply_layer/main.py"

with open(TARGET, "r") as f:
    lines = f.readlines()

# Find PARTIAL_50 in step builder (after the loop that checks action==)
for i, line in enumerate(lines):
    if '"PARTIAL_50"' in line and "action ==" in line:
        start = max(0, i-1)
        end = min(len(lines), i+15)
        print(f">> Found at line {i+1}")
        for j in range(start, end):
            # Print with repr to show exact whitespace
            print(f"LINE {j+1}: {repr(lines[j])}")
        print("---")
