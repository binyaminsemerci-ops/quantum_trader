#!/usr/bin/env python3
"""Find how scanner generates entry signals"""
import os, subprocess

# 1. Find scanner file
for root, dirs, files in os.walk("microservices/autonomous_trader"):
    for f in files:
        if 'scan' in f.lower() and f.endswith('.py'):
            path = os.path.join(root, f)
            print(f"=== {path} ===")
            with open(path) as fh:
                code = fh.read()
            # Find key lines
            for i, line in enumerate(code.splitlines(), 1):
                if any(w in line.lower() for w in ['ai_engine', 'http', 'confidence', 'opportunity', 
                                                    'no entry', 'signal', 'asyncclient', 'post(']):
                    print(f"  {i:4d}: {line.strip()}")

# 2. Find "No entry opportunities" string
r2 = subprocess.run(
    ['grep', '-rn', 'No entry opportunities', 
     'microservices/autonomous_trader/'],
    capture_output=True, text=True
)
print(f"\n'No entry opportunities' location:\n{r2.stdout}")

# 3. Check main.py scanner call
main_path = "microservices/autonomous_trader/main.py"
if os.path.exists(main_path):
    with open(main_path) as f:
        code = f.read()
    for i, line in enumerate(code.splitlines(), 1):
        if any(w in line.lower() for w in ['scanner', 'scan_for', 'entry', 'opportunity']):
            print(f"  main.py {i:4d}: {line.strip()}")
