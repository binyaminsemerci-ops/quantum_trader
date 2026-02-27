#!/usr/bin/env python3
"""Enable PIPELINE_DIAG via env override, restart, wait, read log"""
import subprocess, time

# Add PIPELINE_DIAG override to quantum-execution service
# Use systemd drop-in or direct env override file
override_dir = "/etc/systemd/system/quantum-execution.service.d"
override_file = f"{override_dir}/diag.conf"

import os
os.makedirs(override_dir, exist_ok=True)
with open(override_file, "w") as f:
    f.write("[Service]\nEnvironment=PIPELINE_DIAG=true\n")

subprocess.run(["systemctl", "daemon-reload"])
subprocess.run(["systemctl", "restart", "quantum-execution"])
print("Restarted with PIPELINE_DIAG=true, waiting 75s...")
time.sleep(75)

# Read log for TradeIntent + DIAG lines
with open("/var/log/quantum/execution.log") as f:
    lines = f.readlines()

relevant = []
for line in lines[-500:]:
    if any(k in line for k in [
        "TradeIntent", "DIAG", "EXEC CLOSE", "ACK-INFO", "ACK-BLOCKED",
        "ACK SKIP", "action", "confidence"
    ]):
        relevant.append(line.strip())

print(f"Relevant lines in last 500: {len(relevant)}")
for line in relevant[-50:]:
    print(f"  {line[-200:]}")

# Remove override after done
os.unlink(override_file)
subprocess.run(["systemctl", "daemon-reload"])
print("\n✅ Removed PIPELINE_DIAG override")
