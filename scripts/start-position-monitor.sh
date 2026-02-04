#!/bin/bash
# Start Position Monitor service in background

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "[POSITION MONITOR] Starting standalone position monitor..."
echo "[EXIT BRAIN V3] Exit Brain v3 integration: ENABLED"

# Set environment
export EXIT_BRAIN_V3_ENABLED=true
export POSITION_CHECK_INTERVAL=10

# Start position monitor
python -m microservices.position_monitor.main 2>&1 | tee logs/position_monitor.log &

MONITOR_PID=$!
echo "[POSITION MONITOR] Started with PID: $MONITOR_PID"
echo $MONITOR_PID > /tmp/position_monitor.pid

echo "✅ Position Monitor running (check logs/position_monitor.log)"
