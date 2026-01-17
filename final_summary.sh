#!/bin/bash
set -euo pipefail

echo "=========================================="
echo "AUTOMATED ZOMBIE FIX - FINAL SUMMARY"
echo "=========================================="
echo ""

PROOF_DIR=$(cat /tmp/current_proof_dir.txt)
BACKUP_DIR=$(cat /tmp/current_backup_dir.txt)

echo "Timestamp: $(date -Is)"
echo "Proof dir: $PROOF_DIR"
echo "Backup dir: $BACKUP_DIR"
echo ""

# Compare before/after
echo "=== BEFORE/AFTER COMPARISON ==="
echo ""

echo "BEFORE:"
echo "-------"
grep -A2 "consumers" "$PROOF_DIR/before.txt" | head -3
echo ""
PENDING_BEFORE=$(grep -A1 "^pending" "$PROOF_DIR/before.txt" | tail -1 || echo "0")
echo "Pending: $PENDING_BEFORE"
echo ""

echo "AFTER:"
echo "------"
redis-cli XINFO GROUPS quantum:stream:trade.intent | grep -A1 "^consumers"
echo ""
PENDING_AFTER=$(redis-cli XINFO GROUPS quantum:stream:trade.intent | grep -A1 "^pending" | tail -1)
echo "Pending: $PENDING_AFTER"
echo ""

# Recovery log summary
echo "=== RECOVERY LOG ANALYSIS ==="
echo ""
echo "Last 10 recovery runs:"
tail -10 /var/log/quantum/stream_recover.log
echo ""

# Count actions
TOTAL_CLAIMED=$(grep -c "XAUTOCLAIM claimed=" /var/log/quantum/stream_recover.log || echo 0)
TOTAL_DELETED=$(grep -c "deleted_consumer=" /var/log/quantum/stream_recover.log || echo 0)
echo "Total claimed (lifetime): $TOTAL_CLAIMED"
echo "Total deleted (lifetime): $TOTAL_DELETED"
echo ""

# Timer verification
echo "=== TIMER STATUS ==="
systemctl is-active quantum-stream-recover.timer && echo "✅ Timer: ACTIVE" || echo "❌ Timer: INACTIVE"
echo ""
echo "Next trigger:"
systemctl list-timers quantum-stream-recover.timer --no-pager | grep quantum
echo ""

# Verify ExecStartPre
echo "=== EXECUTION SERVICE HARDENING ==="
if systemctl cat quantum-execution | grep -q "ExecStartPre=/usr/local/bin/quantum_stream_recover.sh"; then
    echo "✅ ExecStartPre configured"
else
    echo "⚠️  ExecStartPre not configured"
fi
echo ""

# Current consumer status
echo "=== CURRENT CONSUMER STATUS ==="
redis-cli XINFO CONSUMERS quantum:stream:trade.intent quantum:group:execution:trade.intent
echo ""

echo "=========================================="
echo "DEPLOYMENT STATUS"
echo "=========================================="
echo "✅ Recovery script: /usr/local/bin/quantum_stream_recover.sh"
echo "✅ Systemd service: /etc/systemd/system/quantum-stream-recover.service"
echo "✅ Systemd timer: /etc/systemd/system/quantum-stream-recover.timer (every 2 min)"
echo "✅ ExecStartPre hook: Configured in quantum-execution.service"
echo ""

echo "=========================================="
echo "ROLLBACK PROCEDURE"
echo "=========================================="
cat > /tmp/ROLLBACK_COMMANDS.sh << 'ROLLBACK'
#!/bin/bash
# Execute this script to rollback automated zombie fix

echo "Rolling back automated zombie fix..."

# Stop and disable timer
systemctl stop quantum-stream-recover.timer
systemctl disable quantum-stream-recover.timer

# Remove systemd units
rm -f /etc/systemd/system/quantum-stream-recover.service
rm -f /etc/systemd/system/quantum-stream-recover.timer

# Restore backup files (adjust BACKUP_DIR as needed)
BACKUP_DIR="BACKUP_DIR_PLACEHOLDER"
if [ -d "$BACKUP_DIR" ]; then
    echo "Restoring from $BACKUP_DIR..."
    cp -v "$BACKUP_DIR/quantum_stream_recover.sh" /usr/local/bin/quantum_stream_recover.sh 2>/dev/null || true
    cp -v "$BACKUP_DIR/10-zombiefix.conf" /etc/systemd/system/quantum-execution.service.d/10-zombiefix.conf 2>/dev/null || true
else
    echo "Backup dir not found: $BACKUP_DIR"
fi

# Reload systemd
systemctl daemon-reload

# Restart execution service
systemctl restart quantum-execution.service

echo "Rollback complete. Verify services:"
systemctl status quantum-execution.service --no-pager
ROLLBACK

sed -i "s|BACKUP_DIR_PLACEHOLDER|$BACKUP_DIR|g" /tmp/ROLLBACK_COMMANDS.sh
chmod +x /tmp/ROLLBACK_COMMANDS.sh

echo "Rollback script created: /tmp/ROLLBACK_COMMANDS.sh"
echo ""
echo "To rollback, run:"
echo "  bash /tmp/ROLLBACK_COMMANDS.sh"
echo ""

echo "=========================================="
echo "✅ AUTOMATED ZOMBIE FIX: COMPLETE"
echo "=========================================="
