#!/bin/bash
set -e

echo '=== AdaptiveLeverageEngine VPS Deployment ==='
echo ''

# Step 1: Backup existing files
echo '[1/6] Backing up existing files...'
cd ~/quantum_trader
cp microservices/exitbrain_v3_5/exit_brain.py microservices/exitbrain_v3_5/exit_brain.py.backup 2>/dev/null || true
cp microservices/exitbrain_v3_5/adaptive_leverage_engine.py microservices/exitbrain_v3_5/adaptive_leverage_engine.py.backup 2>/dev/null || true

# Step 2: Copy new files from /tmp
echo '[2/6] Copying new files...'
cp /tmp/exit_brain.py microservices/exitbrain_v3_5/exit_brain.py
cp /tmp/adaptive_leverage_config.py microservices/exitbrain_v3_5/adaptive_leverage_config.py
cp /tmp/monitor_adaptive_leverage.py .
cp /tmp/ADAPTIVE_LEVERAGE_USAGE_GUIDE.md .

# Step 3: Set permissions
echo '[3/6] Setting permissions...'
chmod 644 microservices/exitbrain_v3_5/exit_brain.py
chmod 644 microservices/exitbrain_v3_5/adaptive_leverage_config.py
chmod 755 monitor_adaptive_leverage.py

# Step 4: Validate Python imports
echo '[4/6] Validating imports...'
cd ~/quantum_trader
python3 << 'PYEOF'
from microservices.exitbrain_v3_5.adaptive_leverage_engine import AdaptiveLeverageEngine
from microservices.exitbrain_v3_5.adaptive_leverage_config import get_config
print('✅ Imports successful')
PYEOF

# Step 5: Run validation script
echo '[5/6] Running validation...'
if [ -f validate_adaptive_leverage.sh ]; then
    bash validate_adaptive_leverage.sh
else
    echo '⚠️ Validation script not found, skipping...'
fi

# Step 6: Show restart instructions
echo '[6/6] Deployment complete!'
echo ''
echo '=== Next Steps ==='
echo 'To restart ExitBrain v3 service:'
echo '  sudo systemctl restart exitbrain_v3'
echo ''
echo 'To monitor adaptive levels:'
echo '  cd ~/quantum_trader'
echo '  python3 monitor_adaptive_leverage.py watch'
echo ''
echo '✅ DEPLOYMENT READY'
