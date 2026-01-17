#!/bin/bash
#
# harvest_brain_rollback.sh - Rollback HarvestBrain microservice
# Usage: bash harvest_brain_rollback.sh
#

set -e

echo "[*] HarvestBrain Rollback Script"
echo "[!] This will stop and disable the quantum-harvest-brain service"
echo ""

# Confirm
read -p "Continue? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Rollback cancelled."
    exit 0
fi

echo ""
echo "=== Stopping Service ==="
if systemctl is-active --quiet quantum-harvest-brain; then
    echo "Stopping quantum-harvest-brain..."
    sudo systemctl stop quantum-harvest-brain
    echo "✅ Service stopped"
else
    echo "⚠️  Service not running"
fi
echo ""

echo "=== Disabling Service ==="
echo "Disabling quantum-harvest-brain from auto-start..."
sudo systemctl disable quantum-harvest-brain 2>/dev/null || true
echo "✅ Service disabled"
echo ""

echo "=== Cleanup (Optional) ==="
echo "Keeping config files in place (in case of re-enable):"
echo "  - /etc/quantum/harvest-brain.env"
echo "  - /etc/systemd/system/quantum-harvest-brain.service"
echo ""
echo "To remove config:"
echo "  sudo rm /etc/quantum/harvest-brain.env"
echo "  sudo rm /etc/systemd/system/quantum-harvest-brain.service"
echo "  sudo systemctl daemon-reload"
echo ""

echo "=== Dedup Keys (Cleanup Optional) ==="
dedup_count=$(redis-cli --no-auth-warning KEYS "quantum:dedup:harvest:*" | wc -l)
if [[ $dedup_count -gt 0 ]]; then
    echo "Found $dedup_count dedup keys"
    echo "To remove: redis-cli --no-auth-warning DEL \$(redis-cli --no-auth-warning KEYS 'quantum:dedup:harvest:*')"
else
    echo "No dedup keys found"
fi
echo ""

echo "✅ Rollback completed"
echo ""
echo "Service Status:"
systemctl is-active quantum-harvest-brain || echo "  (not active)"
