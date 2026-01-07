#!/bin/bash
# Restart a single Quantum Trader service safely
# Usage: ./restart_service.sh <service-name>

set -euo pipefail

SERVICE=$1

if [ -z "$SERVICE" ]; then
    echo "Usage: $0 <service-name>"
    echo "Example: $0 ai-engine"
    exit 1
fi

FULL_NAME="quantum-$SERVICE.service"

echo "🔄 Restarting $FULL_NAME..."
systemctl restart "$FULL_NAME"

echo "⏳ Waiting 5 seconds..."
sleep 5

if systemctl is-active --quiet "$FULL_NAME"; then
    echo "✅ $FULL_NAME is running"
    echo ""
    echo "📋 Recent logs:"
    journalctl -u "$FULL_NAME" -n 20 --no-pager
else
    echo "❌ $FULL_NAME failed to start"
    echo ""
    echo "📋 Error logs:"
    journalctl -u "$FULL_NAME" -n 50 --no-pager
    exit 1
fi
