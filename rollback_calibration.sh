#!/bin/bash
#
# Atomic Calibration Rollback Script
# Reverts calibration config to pre-calibration state
# Restarts affected services
#
# Usage: ./rollback_calibration.sh [reason]
#

set -e  # Exit on error

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_CONFIG="/root/quantum_trader/config/calibration_backup.json"
ACTIVE_CONFIG="/root/quantum_trader/config/calibration_config.json"
ROLLBACK_LOG="/root/logs/rollback_${TIMESTAMP}.log"

echo "================================================" | tee -a "$ROLLBACK_LOG"
echo "🚨 CALIBRATION ROLLBACK INITIATED" | tee -a "$ROLLBACK_LOG"
echo "Time: $(date)" | tee -a "$ROLLBACK_LOG"
echo "Reason: ${1:-Manual rollback}" | tee -a "$ROLLBACK_LOG"
echo "================================================" | tee -a "$ROLLBACK_LOG"

# Step 1: Verify backup exists
if [ ! -f "$BACKUP_CONFIG" ]; then
    echo "❌ ERROR: Backup config not found at $BACKUP_CONFIG" | tee -a "$ROLLBACK_LOG"
    echo "Cannot proceed with rollback!" | tee -a "$ROLLBACK_LOG"
    exit 1
fi

echo "✅ Backup config found" | tee -a "$ROLLBACK_LOG"

# Step 2: Stop affected services
echo "" | tee -a "$ROLLBACK_LOG"
echo "Stopping services..." | tee -a "$ROLLBACK_LOG"
systemctl stop quantum-ai-engine 2>&1 | tee -a "$ROLLBACK_LOG"
systemctl stop quantum-harvest-consumer 2>&1 | tee -a "$ROLLBACK_LOG"
echo "✅ Services stopped" | tee -a "$ROLLBACK_LOG"

# Step 3: Archive current (post-calibration) config
if [ -f "$ACTIVE_CONFIG" ]; then
    ARCHIVE_FILE="/root/quantum_trader/config/calibration_FAILED_${TIMESTAMP}.json"
    cp "$ACTIVE_CONFIG" "$ARCHIVE_FILE"
    echo "✅ Current config archived to: $ARCHIVE_FILE" | tee -a "$ROLLBACK_LOG"
fi

# Step 4: Restore backup config
echo "" | tee -a "$ROLLBACK_LOG"
echo "Restoring pre-calibration config..." | tee -a "$ROLLBACK_LOG"
cp "$BACKUP_CONFIG" "$ACTIVE_CONFIG"
echo "✅ Config restored from backup" | tee -a "$ROLLBACK_LOG"

# Step 5: Clear calibration status in Redis
echo "" | tee -a "$ROLLBACK_LOG"
echo "Clearing calibration status in Redis..." | tee -a "$ROLLBACK_LOG"
redis-cli DEL quantum:calibration:status 2>&1 | tee -a "$ROLLBACK_LOG"
redis-cli SET quantum:calibration:rollback "{\"timestamp\": \"$(date -Iseconds)\", \"reason\": \"${1:-Manual rollback}\"}" 2>&1 | tee -a "$ROLLBACK_LOG"
echo "✅ Redis state cleared" | tee -a "$ROLLBACK_LOG"

# Step 6: Restart services
echo "" | tee -a "$ROLLBACK_LOG"
echo "Restarting services..." | tee -a "$ROLLBACK_LOG"
systemctl start quantum-ai-engine 2>&1 | tee -a "$ROLLBACK_LOG"
sleep 5
systemctl start quantum-harvest-consumer 2>&1 | tee -a "$ROLLBACK_LOG"
sleep 3
echo "✅ Services restarted" | tee -a "$ROLLBACK_LOG"

# Step 7: Verify services are running
echo "" | tee -a "$ROLLBACK_LOG"
echo "Verifying service status..." | tee -a "$ROLLBACK_LOG"

AI_ENGINE_STATUS=$(systemctl is-active quantum-ai-engine)
HARVEST_STATUS=$(systemctl is-active quantum-harvest-consumer)

if [ "$AI_ENGINE_STATUS" = "active" ]; then
    echo "✅ quantum-ai-engine: active" | tee -a "$ROLLBACK_LOG"
else
    echo "❌ quantum-ai-engine: $AI_ENGINE_STATUS" | tee -a "$ROLLBACK_LOG"
fi

if [ "$HARVEST_STATUS" = "active" ]; then
    echo "✅ quantum-harvest-consumer: active" | tee -a "$ROLLBACK_LOG"
else
    echo "❌ quantum-harvest-consumer: $HARVEST_STATUS" | tee -a "$ROLLBACK_LOG"
fi

# Step 8: Wait and verify decisions flowing
echo "" | tee -a "$ROLLBACK_LOG"
echo "Waiting 15 seconds for system stabilization..." | tee -a "$ROLLBACK_LOG"
sleep 15

LATEST_DECISION=$(redis-cli GET quantum:decision:latest 2>/dev/null)
if [ -n "$LATEST_DECISION" ]; then
    DECISION_TS=$(echo "$LATEST_DECISION" | jq -r '.timestamp // empty' 2>/dev/null)
    if [ -n "$DECISION_TS" ]; then
        echo "✅ Decisions flowing (latest: $DECISION_TS)" | tee -a "$ROLLBACK_LOG"
    else
        echo "⚠️  Decision data present but no timestamp" | tee -a "$ROLLBACK_LOG"
    fi
else
    echo "⚠️  No recent decision found (may take time to populate)" | tee -a "$ROLLBACK_LOG"
fi

# Step 9: Summary
echo "" | tee -a "$ROLLBACK_LOG"
echo "================================================" | tee -a "$ROLLBACK_LOG"
echo "✅ ROLLBACK COMPLETE" | tee -a "$ROLLBACK_LOG"
echo "================================================" | tee -a "$ROLLBACK_LOG"
echo "" | tee -a "$ROLLBACK_LOG"
echo "System reverted to pre-calibration state." | tee -a "$ROLLBACK_LOG"
echo "Monitor closely for next 30 minutes." | tee -a "$ROLLBACK_LOG"
echo "" | tee -a "$ROLLBACK_LOG"
echo "Log saved to: $ROLLBACK_LOG" | tee -a "$ROLLBACK_LOG"
echo "Failed config archived to: $ARCHIVE_FILE" | tee -a "$ROLLBACK_LOG"
echo "" | tee -a "$ROLLBACK_LOG"

# Step 10: Post-rollback monitoring reminder
echo "📋 NEXT STEPS:" | tee -a "$ROLLBACK_LOG"
echo "1. Follow FASE A monitoring (30 min)" | tee -a "$ROLLBACK_LOG"
echo "2. Review /root/logs/post_calibration_monitor.log" | tee -a "$ROLLBACK_LOG"
echo "3. Analyze what triggered rollback" | tee -a "$ROLLBACK_LOG"
echo "4. Document findings before re-attempting" | tee -a "$ROLLBACK_LOG"
echo "" | tee -a "$ROLLBACK_LOG"

exit 0
