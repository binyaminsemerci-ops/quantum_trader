#!/bin/bash
# Comprehensive P3 Permit Wait-Loop Verification
# Tests atomic permit consumption and monitors live execution

set -e

SSH_KEY="$HOME/.ssh/hetzner_fresh"
VPS="root@46.224.116.254"
MONITOR_DURATION=120

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  P3 PERMIT WAIT-LOOP VERIFICATION SCRIPT                   ║"
echo "║  Deployed: 2026-01-25 00:36:20 UTC                         ║"
echo "║  Status: Ready to validate atomic permit consumption       ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Verify patch deployment
echo "STEP 1: Verify Patch Deployment"
echo "================================"
wsl ssh -i $SSH_KEY $VPS 'grep -c "wait_and_consume_permits" /root/quantum_trader/microservices/apply_layer/main.py' | \
  awk '{if($1>0) print "✓ Patch present in apply_layer/main.py"; else print "✗ ERROR: Patch missing"}' && \

echo ""

# Step 2: Verify configuration
echo "STEP 2: Verify Environment Configuration"
echo "========================================"
wsl ssh -i $SSH_KEY $VPS 'echo "Config in apply-layer.env:" && grep -E "PERMIT_WAIT|PERMIT_STEP" /etc/quantum/apply-layer.env && echo "✓ Configuration set correctly"' && \
echo ""

# Step 3: Service status
echo "STEP 3: Service Status"
echo "======================"
wsl ssh -i $SSH_KEY $VPS 'systemctl is-active quantum-apply-layer' | \
  awk '{if($1=="active") print "✓ Service running"; else print "✗ Service NOT running"}' && \
echo ""

# Step 4: Check for existing [PERMIT_WAIT] logs
echo "STEP 4: Check Recent PERMIT_WAIT Logs"
echo "====================================="
PERMIT_LOGS=$(wsl ssh -i $SSH_KEY $VPS 'journalctl -u quantum-apply-layer --since "30 minutes ago" --no-pager | grep "\[PERMIT_WAIT\]" | wc -l')
if [ "$PERMIT_LOGS" -gt 0 ]; then
  echo "✓ Found $PERMIT_LOGS existing [PERMIT_WAIT] log entries"
  echo ""
  echo "Recent entries:"
  wsl ssh -i $SSH_KEY $VPS 'journalctl -u quantum-apply-layer --since "30 minutes ago" --no-pager | grep "\[PERMIT_WAIT\]" | tail -5'
else
  echo "⏳ No [PERMIT_WAIT] logs yet (awaiting EXECUTE plan)"
fi
echo ""

# Step 5: Check Redis permit keys
echo "STEP 5: Check Redis Permit Keys"
echo "==============================="
wsl ssh -i $SSH_KEY $VPS 'redis-cli --scan --pattern "quantum:permit:*" | head -10' | \
  awk 'NR==1 {print "Sample permit keys in Redis:"} {print "  " $0}' && \
echo ""

# Step 6: Monitor for new EXECUTE plans
echo "STEP 6: Live Monitoring (${MONITOR_DURATION}s)"
echo "=============================================="
echo "Watching for:"
echo "  • New EXECUTE plans (Plan ... published)"
echo "  • Governor permits (quantum:permit: keys)"
echo "  • [PERMIT_WAIT] logs (atomic consumption)"
echo "  • Order execution logs (Order ... executed)"
echo ""
echo "Monitoring..."

# Set up background monitoring
wsl bash -c "
ssh -i $SSH_KEY $VPS '
  STOP_MARKER=\"/tmp/permit_monitor_stop_\$\$\"
  trap \"touch \$STOP_MARKER\" TERM INT

  # Monitor for plans, permits, and execution
  (
    timeout $MONITOR_DURATION journalctl -u quantum-apply-layer -f --no-pager | grep -E \"Plan .* published|\\[PERMIT_WAIT\\]|Order .* executed\" | while read line; do
      echo \"[$(date -u +\"%H:%M:%S\")] \$line\"
    done
  ) &
  MONITOR_PID=\$!

  # Wait for timeout
  sleep $MONITOR_DURATION

  # Kill monitor
  kill \$MONITOR_PID 2>/dev/null || true
  rm -f \$STOP_MARKER

  # Final summary
  echo \"\"
  echo \"MONITORING COMPLETE\"
  echo \"==================\"
  
  # Count occurrences
  PLANS=\$(journalctl -u quantum-apply-layer --since \"$MONITOR_DURATION seconds ago\" --no-pager | grep \"Plan .* published\" | wc -l)
  PERMITS=\$(journalctl -u quantum-apply-layer --since \"$MONITOR_DURATION seconds ago\" --no-pager | grep \"\\[PERMIT_WAIT\\]\" | wc -l)
  ORDERS=\$(journalctl -u quantum-apply-layer --since \"$MONITOR_DURATION seconds ago\" --no-pager | grep \"Order .* executed\" | wc -l)
  
  echo \"Plans published: \$PLANS\"
  echo \"Permit waits: \$PERMITS\"
  echo \"Orders executed: \$ORDERS\"
'
" || true

echo ""

# Step 7: Performance summary
echo "STEP 7: Permit Wait Performance"
echo "=============================="
wsl ssh -i $SSH_KEY $VPS 'journalctl -u quantum-apply-layer --since "1 hour ago" --no-pager | grep "\[PERMIT_WAIT\] OK" | awk -F"wait_ms=" "{print \$2}" | awk -F" " "{sum+=\$1; count++} END {if(count>0) printf \"Average wait: %.0fms (n=%d)\", sum/count, count; else print \"No OK permits yet\"}' && \
echo ""
echo ""

# Step 8: Error check
echo "STEP 8: Error Summary"
echo "===================="
ERRORS=$(wsl ssh -i $SSH_KEY $VPS 'journalctl -u quantum-apply-layer --since "1 hour ago" --no-pager | grep -E "error|ERROR|FATAL" | wc -l')
if [ "$ERRORS" -eq 0 ]; then
  echo "✓ No errors in logs"
else
  echo "⚠ Found $ERRORS error entries:"
  wsl ssh -i $SSH_KEY $VPS 'journalctl -u quantum-apply-layer --since "1 hour ago" --no-pager | grep -E "error|ERROR|FATAL" | head -5'
fi
echo ""

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  VERIFICATION COMPLETE                                     ║"
echo "║                                                            ║"
echo "║  Next Steps:                                               ║"
echo "║  1. If PERMIT_WAIT logs appear → patch is working         ║"
echo "║  2. If no logs → wait for next EXECUTE plan to trigger    ║"
echo "║  3. Check order execution in next EXECUTE                 ║"
echo "║                                                            ║"
echo "║  To force fresh EXECUTE:                                  ║"
echo "║  redis-cli DEL quantum:apply:*                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
