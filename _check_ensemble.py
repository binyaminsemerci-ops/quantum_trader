#!/usr/bin/env python3
"""Check quantum-ai-engine health and ensemble model status"""
import subprocess
import json
import sys

# Health check
r = subprocess.run(["curl", "-s", "http://127.0.0.1:8001/health"], 
                   capture_output=True, text=True, timeout=10)

print("=== HEALTH RESPONSE ===")
if r.stdout:
    try:
        data = json.loads(r.stdout)
        print(json.dumps(data, indent=2)[:3000])
    except:
        print(r.stdout[:2000])
else:
    print("NO RESPONSE")
    print("stderr:", r.stderr[:500])

# Check ensemble manager state
print("\n=== ENSEMBLE STATUS (from logs) ===")
r2 = subprocess.run(
    ["journalctl", "-u", "quantum-ai-engine", "-n", "500", "--no-pager"],
    capture_output=True, text=True
)
lines = r2.stdout.split("\n")
keywords = ["xgb", "lgbm", "dlinear", "nhits", "patchtst", "tft", 
            "Ensemble", "DEPLOY-DEBUG", "XGB-Agent", "DLinear-Agent",
            "Loaded", "loaded", "ensemble_manager", "ERROR", "failed",
            "prediction failed", "NoneType"]
for line in lines:
    for kw in keywords:
        if kw.lower() in line.lower():
            print(line[-300:])
            break

# Check which models loaded
print("\n=== AGENT LOADING CHECK ===")
r3 = subprocess.run(["systemctl", "is-active", "quantum-ai-engine"],
                   capture_output=True, text=True)
print("Service status:", r3.stdout.strip())
