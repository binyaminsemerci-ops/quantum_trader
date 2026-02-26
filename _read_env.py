#!/usr/bin/env python3
"""Read all relevant env files and show non-secret values"""
import subprocess

files = [
    "/etc/quantum/intent-executor.env",
    "/etc/quantum/execution.env",
    "/etc/quantum/quantum.env",
    "/etc/quantum/main.env",
]

for f in files:
    try:
        with open(f) as fh:
            lines = fh.readlines()
        print(f"\n=== {f} ===")
        for line in lines:
            lstrip = line.strip()
            if not lstrip or lstrip.startswith("#"):
                print(f"  {lstrip}")
                continue
            k = lstrip.split("=")[0]
            v = lstrip.split("=", 1)[1] if "=" in lstrip else ""
            # Mask secrets
            if any(x in k.upper() for x in ["SECRET", "PASSWORD", "PASS", "TOKEN"]):
                print(f"  {k}=***MASKED***")
            elif any(x in k.upper() for x in ["API_KEY", "APIKEY"]):
                print(f"  {k}={v[:8]}***")
            else:
                print(f"  {lstrip}")
    except FileNotFoundError:
        print(f"\n=== {f} === NOT FOUND")
    except Exception as e:
        print(f"\n=== {f} === ERROR: {e}")

# List all etc/quantum files
print("\n=== /etc/quantum/ contents ===")
try:
    result = subprocess.run(["ls", "-la", "/etc/quantum/"], capture_output=True, text=True)
    print(result.stdout)
except:
    pass
