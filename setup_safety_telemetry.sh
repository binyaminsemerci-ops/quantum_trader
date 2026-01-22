#!/bin/bash
set -e

echo "=== PHASE 1: SETUP VENV ==="
mkdir -p /opt/quantum/venvs/safety-telemetry
python3 -m venv /opt/quantum/venvs/safety-telemetry
/opt/quantum/venvs/safety-telemetry/bin/pip install --quiet --upgrade pip
/opt/quantum/venvs/safety-telemetry/bin/pip install --quiet redis prometheus_client

echo ""
echo "Installed packages:"
/opt/quantum/venvs/safety-telemetry/bin/pip list | grep -E "redis|prometheus"

echo ""
echo "=== PHASE 2: CREATE DIRECTORIES ==="
mkdir -p /home/qt/quantum_trader/microservices/safety_telemetry
mkdir -p /home/qt/quantum_trader/grafana/dashboards
mkdir -p /etc/quantum

echo "✅ Setup complete"
