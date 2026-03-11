#!/usr/bin/env python3
"""Fix quantum-ai-engine service and env file to use /home/qt/quantum_trader instead of /opt/quantum"""

# Fix service file
with open("/etc/systemd/system/quantum-ai-engine.service", "r") as f:
    content = f.read()

content = content.replace("/opt/quantum/venvs/ai-engine/bin/python -m uvicorn",
                          "/home/qt/quantum_trader_venv/bin/uvicorn")
content = content.replace("WorkingDirectory=/opt/quantum",
                          "WorkingDirectory=/home/qt/quantum_trader")
content = content.replace('PYTHONPATH=/opt/quantum"',
                          'PYTHONPATH=/home/qt/quantum_trader"')

with open("/etc/systemd/system/quantum-ai-engine.service", "w") as f:
    f.write(content)
print("SERVICE_UPDATED_OK")

# Fix env file
with open("/etc/quantum/ai-engine.env", "r") as f:
    env = f.read()

env = env.replace("/opt/quantum/ai_engine/models/",
                  "/home/qt/quantum_trader/ai_engine/models/")

with open("/etc/quantum/ai-engine.env", "w") as f:
    f.write(env)
print("ENV_UPDATED_OK")

# Verify
print("\n--- Service ExecStart ---")
for line in open("/etc/systemd/system/quantum-ai-engine.service"):
    if any(k in line for k in ["WorkingDirectory", "ExecStart", "PYTHONPATH"]):
        print(line.rstrip())

print("\n--- Env model paths ---")
for line in open("/etc/quantum/ai-engine.env"):
    if "MODEL_PATH" in line or "SCALER_PATH" in line:
        print(line.rstrip())
