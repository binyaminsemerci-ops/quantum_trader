#!/bin/bash
# Get last startup block in rl_trainer.err
echo "=== Last 30 lines of err log ==="
tail -30 /opt/quantum/logs/rl_trainer.err

echo ""
echo "=== Last Errno 98 occurrences ==="
grep -c "Errno 98" /opt/quantum/logs/rl_trainer.err && echo "still happening" || echo "GONE"
grep "Errno 98" /opt/quantum/logs/rl_trainer.err | tail -3

echo ""
echo "=== PORT env check ==="
grep PORT /opt/quantum/.env
python3 -c "
import os, sys
sys.path.insert(0, '/home/qt/quantum_trader')
os.chdir('/home/qt/quantum_trader')
# activate venv env vars manually
os.environ['VIRTUAL_ENV'] = '/opt/quantum/venvs/runtime'
# Load the .env file manually
with open('/opt/quantum/.env') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            os.environ.setdefault(k.strip(), v.strip())
from microservices.rl_training.config import settings
print(f'settings.PORT = {settings.PORT}')
"

echo ""
echo "=== What's on port 8007 now ==="
ss -tlnp | grep 8007 || echo "8007 is FREE"
echo "=== What's on port 8005 now ==="
ss -tlnp | grep 8005 || echo "8005 is FREE"
