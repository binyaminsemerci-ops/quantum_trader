#!/bin/bash
# Phase 2 Harvest Brain Verification
# Output saved to file for inspection

OUTPUT_FILE="/tmp/phase2_verification_$(date +%s).txt"

echo "=== PHASE 2 VERIFICATION $(date -u) ===" > $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "1. Service Status:" >> $OUTPUT_FILE
systemctl status quantum-harvest-brain --no-pager -l >> $OUTPUT_FILE 2>&1
echo "" >> $OUTPUT_FILE

echo "2. Is-Active:" >> $OUTPUT_FILE
systemctl is-active quantum-harvest-brain >> $OUTPUT_FILE 2>&1
echo "" >> $OUTPUT_FILE

echo "3. Process Check:" >> $OUTPUT_FILE
ps aux | grep harvest_brain | grep -v grep >> $OUTPUT_FILE 2>&1
echo "" >> $OUTPUT_FILE

echo "4. Recent Logs:" >> $OUTPUT_FILE
journalctl -u quantum-harvest-brain --since "10 minutes ago" --no-pager -n 50 >> $OUTPUT_FILE 2>&1
echo "" >> $OUTPUT_FILE

echo "5. Harvest Stream:" >> $OUTPUT_FILE
redis-cli XLEN quantum:stream:harvest.intent >> $OUTPUT_FILE 2>&1
redis-cli XREVRANGE quantum:stream:harvest.intent + - COUNT 2 >> $OUTPUT_FILE 2>&1
echo "" >> $OUTPUT_FILE

echo "6. Service File:" >> $OUTPUT_FILE
grep ExecStart /etc/systemd/system/quantum-harvest-brain.service >> $OUTPUT_FILE 2>&1
echo "" >> $OUTPUT_FILE

echo "7. Failed Services Count:" >> $OUTPUT_FILE
systemctl list-units --type=service --state=failed | grep quantum | wc -l >> $OUTPUT_FILE 2>&1
echo "" >> $OUTPUT_FILE

echo "=== OUTPUT SAVED TO: $OUTPUT_FILE ===" >> $OUTPUT_FILE

cat $OUTPUT_FILE
