#!/bin/bash
#
# harvest_brain_proof.sh - Verify HarvestBrain microservice
# Usage: bash harvest_brain_proof.sh
#

set -e

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PROOF_DIR="/tmp/phase_e_harvest_brain_$(date +%s)"
mkdir -p "$PROOF_DIR"

echo "[*] HarvestBrain Proof Script"
echo "[*] Proof directory: $PROOF_DIR"
echo ""

# 1. Check if service is running
echo "=== Service Status ==="
if systemctl is-active --quiet quantum-harvest-brain; then
    echo "✅ quantum-harvest-brain is ACTIVE"
    systemctl status quantum-harvest-brain --no-pager | head -15 | tee "$PROOF_DIR/service_status.txt"
else
    echo "❌ quantum-harvest-brain is NOT active"
    echo "Starting service..."
    sudo systemctl start quantum-harvest-brain
    sleep 2
fi
echo ""

# 2. Check config file exists
echo "=== Config File ==="
if [[ -f /etc/quantum/harvest-brain.env ]]; then
    echo "✅ Config found at /etc/quantum/harvest-brain.env"
    grep -v "^#" /etc/quantum/harvest-brain.env | grep -v "^$" > "$PROOF_DIR/config_active.txt"
    cat "$PROOF_DIR/config_active.txt"
else
    echo "❌ Config NOT found at /etc/quantum/harvest-brain.env"
    exit 1
fi
echo ""

# 3. Check consumer group exists
echo "=== Stream Consumer Group ==="
redis-cli --no-auth-warning XINFO GROUPS quantum:stream:execution.result 2>&1 | tee "$PROOF_DIR/consumer_groups.txt"
if grep -q "harvest_brain_group" "$PROOF_DIR/consumer_groups.txt"; then
    echo "✅ Consumer group 'harvest_brain_group' exists"
else
    echo "⚠️  Consumer group may not yet exist (created on first run)"
fi
echo ""

# 4. Check output streams
echo "=== Output Streams ==="
for stream in quantum:stream:harvest.suggestions quantum:stream:trade.intent; do
    count=$(redis-cli --no-auth-warning XLEN "$stream")
    echo "Stream: $stream"
    echo "  Entries: $count"
    if [[ $count -gt 0 ]]; then
        redis-cli --no-auth-warning XREVRANGE "$stream" + - COUNT 2 | tee -a "$PROOF_DIR/streams_$stream.txt"
    fi
done
echo ""

# 5. Check for dedup keys
echo "=== Dedup Keys ==="
dedup_count=$(redis-cli --no-auth-warning KEYS "quantum:dedup:harvest:*" | wc -l)
echo "Active dedup keys: $dedup_count"
redis-cli --no-auth-warning KEYS "quantum:dedup:harvest:*" 2>&1 | head -10 | tee "$PROOF_DIR/dedup_keys.txt"
echo ""

# 6. Check kill-switch
echo "=== Kill-Switch ==="
kill_switch=$(redis-cli --no-auth-warning GET quantum:kill)
if [[ "$kill_switch" == "1" ]]; then
    echo "⚠️  Kill-switch is ACTIVE (no publishing)"
else
    echo "✅ Kill-switch is OFF (publishing enabled)"
fi
echo ""

# 7. Check service logs (last 20 lines)
echo "=== Recent Logs ==="
journalctl -u quantum-harvest-brain -n 20 --no-pager 2>&1 | tee "$PROOF_DIR/logs_recent.txt"
echo ""

# 8. Test mode check
echo "=== Harvest Mode ==="
mode=$(grep "^HARVEST_MODE" /etc/quantum/harvest-brain.env | cut -d'=' -f2)
echo "Current mode: $mode"
echo "  Shadow: No live orders (safe)"
echo "  Live: Publishing to trade.intent"
echo ""

# 9. Memory usage
echo "=== Memory Usage ==="
ps aux | grep harvest_brain | grep -v grep | tee "$PROOF_DIR/process_info.txt"
echo ""

# 10. Summary
echo "=== Summary ==="
echo "Proof artifacts saved to: $PROOF_DIR"
echo ""
echo "Next Steps:"
echo "1. Monitor logs: journalctl -u quantum-harvest-brain -f"
echo "2. Check output streams:"
echo "   redis-cli XREVRANGE quantum:stream:harvest.suggestions + - COUNT 5"
echo "3. Test kill-switch: redis-cli SET quantum:kill 1"
echo "4. When validated, switch to live: HARVEST_MODE=live in /etc/quantum/harvest-brain.env"
echo ""
echo "✅ Proof script completed"
