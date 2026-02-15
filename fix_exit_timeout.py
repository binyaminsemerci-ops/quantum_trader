#!/usr/bin/env python3
"""
Quick script to SSH and fix exit_manager.py on VPS
"""
import subprocess
import sys

# SSH command base
SSH = ["wsl", "ssh", "-i", "~/.ssh/hetzner_fresh", "root@46.224.116.254"]

def run(cmd):
    """Run SSH command and return output"""
    full_cmd = SSH + [cmd]
    result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=60)
    return result.stdout + result.stderr

# Test connection
print("Testing SSH connection...")
out = run("hostname")
print(f"Connected to: {out.strip()}")

# Check current exit_manager state
print("\nChecking exit_manager.py lines 130-145...")
out = run("sed -n '130,145p' /home/qt/quantum_trader/microservices/autonomous_trader/exit_manager.py")
print(out)

# Add local fallback code via Python on VPS
print("\nAdding local fallback code...")
patch_script = '''
import re

filepath = "/home/qt/quantum_trader/microservices/autonomous_trader/exit_manager.py"
with open(filepath, "r") as f:
    content = f.read()

# Find the "# Default: hold" section and add fallback before it
if "LOCAL FALLBACK" not in content:
    old = "        # Default: hold"
    new = """        # LOCAL FALLBACK when AI unavailable - R-based exits
        R = position.R_net
        age_hours = position.age_seconds / 3600 if hasattr(position, "age_seconds") else 0
        
        # Profit taking: R > 2.0 = close all
        if R > 2.0:
            return ExitDecision(
                symbol=position.symbol,
                action="CLOSE",
                percentage=1.0,
                reason=f"local_fallback_profit_R={R:.2f}",
                hold_score=1,
                exit_score=8,
                factors={"R_net": R, "local_fallback": True}
            )
        
        # Profit taking: R > 1.0 = partial close 50%
        if R > 1.0:
            return ExitDecision(
                symbol=position.symbol,
                action="PARTIAL_CLOSE",
                percentage=0.5,
                reason=f"local_fallback_profit_R={R:.2f}",  
                hold_score=3,
                exit_score=5,
                factors={"R_net": R, "local_fallback": True}
            )
        
        # Loss cutting: R < -1.0 and position > 4 hours
        if R < -1.0 and age_hours > 4:
            return ExitDecision(
                symbol=position.symbol,
                action="CLOSE",
                percentage=1.0,
                reason=f"local_fallback_loss_R={R:.2f}",
                hold_score=0,
                exit_score=7,
                factors={"R_net": R, "age_hours": age_hours, "local_fallback": True}
            )

        # Default: hold"""
    
    content = content.replace(old, new)
    
    with open(filepath, "w") as f:
        f.write(content)
    print("PATCHED: Added local fallback")
else:
    print("SKIP: Local fallback already exists")
'''

# Write patch script to VPS
run(f"echo '{patch_script}' > /tmp/patch_exit.py")
out = run("python3 /tmp/patch_exit.py")
print(out)

# Restart autonomous trader
print("\nRestarting autonomous trader...")
out = run("systemctl restart quantum-autonomous-trader && sleep 3 && systemctl is-active quantum-autonomous-trader")
print(f"Autonomous trader: {out.strip()}")

# Check for logs
print("\nWaiting 30s for exit evaluation...")
import time
time.sleep(30)
out = run("journalctl -u quantum-autonomous-trader --since '40 sec ago' 2>/dev/null | grep -i 'exit\\|close\\|fallback' | tail -15")
print(out)

print("\n=== FIX COMPLETE ===")
