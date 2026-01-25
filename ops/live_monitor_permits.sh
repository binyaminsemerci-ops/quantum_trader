#!/bin/bash
# P3 Permit Wait-Loop Live Monitor
# Watches for next EXECUTE plan and validates atomic permit consumption
# Color-coded output, real-time metrics

SSH_KEY="$HOME/.ssh/hetzner_fresh"
VPS="root@46.224.116.254"
TIMEOUT_SEC=120
START_TIME=$(date +%s)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

clear_header() {
  echo ""
  echo "╔════════════════════════════════════════════════════════════════╗"
  echo "║  P3 PERMIT WAIT-LOOP LIVE MONITOR                              ║"
  echo "║  Watching for EXECUTE plans and atomic permit consumption      ║"
  echo "╚════════════════════════════════════════════════════════════════╝"
  echo ""
  echo "Configuration:"
  echo "  Max wait: 1200ms (PERMIT_WAIT_MS)"
  echo "  Poll interval: 100ms (PERMIT_STEP_MS)"
  echo "  Timeout: ${TIMEOUT_SEC}s"
  echo "  Started: $(date -u)"
  echo ""
  echo "─────────────────────────────────────────────────────────────────"
  echo ""
}

elapsed() {
  echo $(($(date +%s) - START_TIME))
}

clear_header

# Start monitoring in background
echo "$(date -u '+%H:%M:%S') Starting monitor..."
echo ""

wsl ssh -i $SSH_KEY $VPS "
  TIMEOUT=$TIMEOUT_SEC
  START_EPOCH=\$(date +%s)

  # Start log stream in background
  journalctl -u quantum-apply-layer -f --no-pager &
  LOG_PID=\$!

  # Monitor loop
  while true; do
    CURRENT_EPOCH=\$(date +%s)
    ELAPSED=\$((CURRENT_EPOCH - START_EPOCH))
    
    if [ \$ELAPSED -ge \$TIMEOUT ]; then
      kill \$LOG_PID 2>/dev/null
      exit 0
    fi
    
    sleep 1
  done
" | while IFS= read -r line; do
  
  ELAPSED=$(elapsed)
  
  # Check for key events
  if echo "$line" | grep -q "Plan .* published.*EXECUTE"; then
    echo -e "${GREEN}[EXECUTE FOUND @${ELAPSED}s]${NC} $line"
  elif echo "$line" | grep -q "\[PERMIT_WAIT\] OK"; then
    # Extract metrics
    PLAN=$(echo "$line" | grep -oP 'plan=\K[a-f0-9]+' | head -1)
    WAIT=$(echo "$line" | grep -oP 'wait_ms=\K[0-9]+' | head -1)
    QTY=$(echo "$line" | grep -oP 'safe_qty=\K[0-9.]+' | head -1)
    echo -e "${GREEN}[PERMIT OK @${ELAPSED}s]${NC} plan=${PLAN:0:8}... wait_ms=${WAIT} safe_qty=${QTY}"
  elif echo "$line" | grep -q "\[PERMIT_WAIT\] BLOCK"; then
    PLAN=$(echo "$line" | grep -oP 'plan=\K[a-f0-9]+' | head -1)
    REASON=$(echo "$line" | grep -oP 'reason=\K[^ ]+' | head -1)
    echo -e "${RED}[PERMIT BLOCKED @${ELAPSED}s]${NC} plan=${PLAN:0:8}... reason=${REASON}"
  elif echo "$line" | grep -q "Order .* executed"; then
    echo -e "${GREEN}[ORDER EXECUTED @${ELAPSED}s]${NC} $line"
  elif echo "$line" | grep -q "error\|ERROR\|FATAL"; then
    echo -e "${RED}[ERROR @${ELAPSED}s]${NC} $line"
  fi
  
  # Stop if timeout reached
  if [ $ELAPSED -ge $TIMEOUT_SEC ]; then
    echo ""
    echo "─────────────────────────────────────────────────────────────────"
    echo -e "${YELLOW}Monitoring timeout reached (${TIMEOUT_SEC}s)${NC}"
    break
  fi
done

echo ""
echo "─────────────────────────────────────────────────────────────────"
echo ""
echo "Monitoring Complete!"
echo "Total elapsed: $(elapsed)s"
echo ""

# Show summary
wsl ssh -i $SSH_KEY $VPS "
  echo 'SUMMARY (Last 30 minutes):' 
  echo ''
  echo 'Plans published:'
  journalctl -u quantum-apply-layer --since '30 minutes ago' --no-pager | grep 'Plan .* published' | wc -l
  
  echo 'Permit wait events:'
  journalctl -u quantum-apply-layer --since '30 minutes ago' --no-pager | grep '\[PERMIT_WAIT\]' | wc -l
  
  echo 'Success (OK):'
  journalctl -u quantum-apply-layer --since '30 minutes ago' --no-pager | grep '\[PERMIT_WAIT\] OK' | wc -l
  
  echo 'Blocked:'
  journalctl -u quantum-apply-layer --since '30 minutes ago' --no-pager | grep '\[PERMIT_WAIT\] BLOCK' | wc -l
  
  echo 'Orders executed:'
  journalctl -u quantum-apply-layer --since '30 minutes ago' --no-pager | grep 'Order .* executed' | wc -l
  
  echo ''
  echo 'Latest PERMIT_WAIT logs:'
  journalctl -u quantum-apply-layer --since '30 minutes ago' --no-pager | grep '\[PERMIT_WAIT\]' | tail -5
" 

echo ""
echo "Next steps:"
echo "1. If OK logs appeared → Patch is working ✓"
echo "2. If no EXECUTE yet → Wait for next trading signal"
echo "3. If BLOCKED → Check permit issuance in Governor/P3.3"
echo "4. Clear cache to force fresh: redis-cli DEL quantum:apply:*"
