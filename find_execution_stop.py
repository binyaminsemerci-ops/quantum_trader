#!/usr/bin/env python3
"""
Find exact line where execution stops after "Direct execution mode"
"""
import subprocess

result = subprocess.run(
    ["docker", "logs", "quantum_backend", "--since", "5m"],
    capture_output=True,
    text=True,
    encoding='utf-8',
    errors='ignore'
)

lines = result.stdout.split('\n') + result.stderr.split('\n')

# Find "Direct execution mode" and show next 50 lines
for i, line in enumerate(lines):
    if "Direct execution mode" in line:
        print(f"Found at line {i}:\n")
        for j in range(i, min(i+50, len(lines))):
            if any(x in lines[j] for x in ["event_driven_executor", "BRIEFCASE", "MONEY", "submit", "order", "skip", "block", "fail"]):
                print(lines[j])
        break
