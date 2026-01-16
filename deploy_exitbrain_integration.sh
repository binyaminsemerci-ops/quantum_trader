#!/bin/bash
# Deploy ExitBrain v3.5 Integration to Position Monitor

set -e
cd ~/quantum_trader

SERVICE="quantum-position-monitor"
SERVICE_DIR=~/quantum_trader/microservices/position_monitor

echo "🔄 Deploying ExitBrain v3.5 Integration..."
echo "========================================="

# 1️⃣ Backup original main.py
if [ -f "$SERVICE_DIR/main.py" ]; then
    echo "📦 Backing up original main.py..."
    cp "$SERVICE_DIR/main.py" "$SERVICE_DIR/main.py.backup"
fi

# 2️⃣ Replace main.py with ExitBrain integrated version
echo "📝 Deploying new main.py with ExitBrain v3.5..."
cp "$SERVICE_DIR/main_exitbrain.py" "$SERVICE_DIR/main.py"

# 3️⃣ Verify exitbrain_v3_5 module path
if [ -d "microservices/exitbrain_v3_5" ]; then
  echo "✅ exitbrain_v3_5 module present."
else
  echo "❌ exitbrain_v3_5 module missing!"
  exit 1
fi

# 4️⃣ Restart position-monitor service
echo "🔁 Restarting position-monitor service..."
sudo systemctl restart $SERVICE.service

# 5️⃣ Wait for service to be healthy
echo "⏳ Waiting 15 seconds for service to restart..."
sleep 15

# 6️⃣ Check logs for ExitBrain initialization
echo ""
echo "📊 Checking ExitBrain v3.5 initialization logs..."
echo "=================================================="
sudo journalctl -u $SERVICE.service -n 30 --no-pager | grep -E "ExitBrain|POSITION MONITOR"

echo ""
echo "✅ Deployment complete!"
echo ""
echo "To verify ExitBrain is active:"
echo "  sudo journalctl -u $SERVICE.service -f | grep ExitBrain"
