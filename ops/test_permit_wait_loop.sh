#!/bin/bash
# P3 Permit Wait-Loop Verification Script
# Tests the new atomic permit consumption logic

set -e

echo "=========================================="
echo "P3 PERMIT WAIT-LOOP TEST"
echo "Testing: Atomic Governor + P3.3 permit consumption"
echo "Started: $(date)"
echo "=========================================="
echo ""

# Config
PLAN_ID=""
MAX_WAIT=120

echo "Step 1: Waiting for next EXECUTE plan..."
echo ""

# Watch Apply Layer logs for EXECUTE
start_ts=$(date +%s)
while true; do
  now_ts=$(date +%s)
  elapsed=$((now_ts - start_ts))
  
  if [ $elapsed -gt $MAX_WAIT ]; then
    echo "❌ Timeout after ${MAX_WAIT}s - no EXECUTE detected"
    exit 1
  fi
  
  # Check for [PERMIT_WAIT] OK or BLOCK in recent logs
  result=$(journalctl -u quantum-apply-layer --since "5 seconds ago" --no-pager | grep "\[PERMIT_WAIT\]" | head -1)
  
  if [ -n "$result" ]; then
    echo "✓ Found permit-wait log:"
    echo "  $result"
    echo ""
    
    # Extract plan_id and status
    if echo "$result" | grep -q "OK"; then
      echo "✓✓ PERMITS CONSUMED SUCCESSFULLY"
      echo "   Status: Atomic consumption worked!"
      exit 0
    elif echo "$result" | grep -q "BLOCK"; then
      echo "❌ PERMITS BLOCKED"
      echo "   Status: See error in log above"
      exit 1
    fi
  fi
  
  # Show progress
  echo -ne "[${elapsed}/${MAX_WAIT}] Waiting for EXECUTE plan...\r"
  sleep 1
done
