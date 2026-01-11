#!/bin/bash
# QSC-Compliant Canary Activation - ONE Model at 10% Traffic
# NO TRAINING. NO AUTO-SCALE. Immediate rollback on violation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Activation params (QSC)
CANARY_MODEL="xgb"           # First model to activate
CANARY_WEIGHT="0.10"         # 10% traffic
CUTOVER_TS="2026-01-10T05:43:15Z"
MIN_EVENTS=200

# Paths
ACTIVATION_LOG="/var/log/quantum/canary_activation.log"
ROLLBACK_CMD_FILE="/var/log/quantum/canary_rollback.sh"
QUALITY_GATE_SCRIPT="$REPO_ROOT/ops/model_safety/quality_gate.py"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
    echo "[$(date -Iseconds)] $*" | tee -a "$ACTIVATION_LOG"
}

error() {
    echo -e "${RED}[$(date -Iseconds)] ERROR: $*${NC}" | tee -a "$ACTIVATION_LOG"
}

success() {
    echo -e "${GREEN}[$(date -Iseconds)] SUCCESS: $*${NC}" | tee -a "$ACTIVATION_LOG"
}

warn() {
    echo -e "${YELLOW}[$(date -Iseconds)] WARNING: $*${NC}" | tee -a "$ACTIVATION_LOG"
}

# Ensure log directory exists
mkdir -p "$(dirname "$ACTIVATION_LOG")"

log "=========================================="
log "CANARY ACTIVATION - QSC MODE"
log "=========================================="
log "Model: $CANARY_MODEL"
log "Weight: $CANARY_WEIGHT (10% traffic)"
log "Cutover: $CUTOVER_TS"
log "Min events: $MIN_EVENTS"
log ""

# Step 1: Run quality gate
log "Step 1: Running quality gate with cutover filter..."
cd "$REPO_ROOT"
source /opt/quantum/venvs/ai-engine/bin/activate

if python3 "$QUALITY_GATE_SCRIPT" --after "$CUTOVER_TS" > /tmp/quality_gate_output.txt 2>&1; then
    GATE_EXIT_CODE=0
    success "Quality gate PASSED (exit 0)"
    cat /tmp/quality_gate_output.txt | tee -a "$ACTIVATION_LOG"
else
    GATE_EXIT_CODE=$?
    error "Quality gate FAILED (exit $GATE_EXIT_CODE)"
    cat /tmp/quality_gate_output.txt | tee -a "$ACTIVATION_LOG"
    log ""
    log "=========================================="
    error "BLOCKER: Quality gate must pass before activation"
    log "=========================================="
    log "Action: Wait for more events or investigate failures"
    exit 2
fi

# Step 2: Check event count
EVENT_COUNT=$(grep "Events analyzed:" /tmp/quality_gate_output.txt | awk '{print $3}')
log ""
log "Step 2: Verifying event count..."
log "Found: $EVENT_COUNT events"
log "Required: $MIN_EVENTS events"

if [ "$EVENT_COUNT" -lt "$MIN_EVENTS" ]; then
    error "BLOCKER: Insufficient events ($EVENT_COUNT < $MIN_EVENTS)"
    log "Action: Wait for $(($MIN_EVENTS - $EVENT_COUNT)) more events"
    exit 2
fi

success "Event count sufficient ($EVENT_COUNT >= $MIN_EVENTS)"
log ""

# Step 3: Verify normalization audit
log "Step 3: Checking confidence normalization..."
VIOLATIONS=$(grep "Violations (BLOCKER):" /tmp/quality_gate_output.txt | awk '{print $3}')
log "Violations found: $VIOLATIONS"

if [ "$VIOLATIONS" != "0" ]; then
    error "BLOCKER: Confidence violations detected ($VIOLATIONS)"
    log "Action: Fix model outputs before activation"
    exit 2
fi

success "No confidence violations (0 found)"
log ""

# Step 4: Check model exists in telemetry
log "Step 4: Verifying canary model in telemetry..."
if ! grep -q "### $CANARY_MODEL" /tmp/quality_gate_output.txt; then
    error "BLOCKER: Model '$CANARY_MODEL' not found in telemetry"
    log "Available models:"
    grep "^### " /tmp/quality_gate_output.txt | sed 's/###/  -/' | tee -a "$ACTIVATION_LOG"
    exit 2
fi

success "Canary model '$CANARY_MODEL' found in telemetry"
log ""

# Step 5: Generate rollback command
log "Step 5: Generating rollback command..."
ACTIVATION_TS=$(date -Iseconds)

cat > "$ROLLBACK_CMD_FILE" << 'ROLLBACK_EOF'
#!/bin/bash
# CANARY ROLLBACK - Immediate deactivation
# Generated: __ACTIVATION_TS__

set -euo pipefail

CANARY_MODEL="__CANARY_MODEL__"
ACTIVATION_TS="__ACTIVATION_TS__"

echo "[$(date -Iseconds)] ROLLBACK: Deactivating canary model '$CANARY_MODEL'"

# Stop AI engine
echo "[$(date -Iseconds)] Stopping AI engine..."
systemctl stop quantum-ai-engine.service

# Remove canary config (restore 0% weight)
CONFIG_FILE="/etc/quantum/ai-engine.env"
if [ -f "${CONFIG_FILE}.pre_canary_backup" ]; then
    echo "[$(date -Iseconds)] Restoring config from backup..."
    cp "${CONFIG_FILE}.pre_canary_backup" "$CONFIG_FILE"
else
    echo "[$(date -Iseconds)] WARNING: No backup found, setting ${CANARY_MODEL}_WEIGHT=0.0"
    sed -i "s/^${CANARY_MODEL}_WEIGHT=.*/${CANARY_MODEL}_WEIGHT=0.0/" "$CONFIG_FILE"
fi

# Restart AI engine
echo "[$(date -Iseconds)] Restarting AI engine..."
systemctl start quantum-ai-engine.service

# Wait for health check
sleep 5
if systemctl is-active --quiet quantum-ai-engine.service; then
    echo "[$(date -Iseconds)] ✅ AI engine restarted successfully"
else
    echo "[$(date -Iseconds)] ❌ AI engine failed to start"
    journalctl -u quantum-ai-engine.service --since "$ACTIVATION_TS" --no-pager | tail -50
    exit 1
fi

# Verify model weight
echo "[$(date -Iseconds)] Verifying model weight..."
if grep -q "^${CANARY_MODEL}_WEIGHT=0.0" "$CONFIG_FILE"; then
    echo "[$(date -Iseconds)] ✅ Model weight reset to 0.0"
else
    echo "[$(date -Iseconds)] ⚠️  WARNING: Could not verify weight reset"
fi

echo ""
echo "=========================================="
echo "✅ ROLLBACK COMPLETE"
echo "=========================================="
echo "Model: $CANARY_MODEL"
echo "Activation: $ACTIVATION_TS"
echo "Rollback: $(date -Iseconds)"
echo "Status: Deactivated (weight 0.0)"
echo ""
echo "Next: Check logs and investigate violation"
echo "  journalctl -u quantum-ai-engine.service --since '$ACTIVATION_TS'"
ROLLBACK_EOF

# Replace placeholders
sed -i "s|__ACTIVATION_TS__|$ACTIVATION_TS|g" "$ROLLBACK_CMD_FILE"
sed -i "s|__CANARY_MODEL__|$CANARY_MODEL|g" "$ROLLBACK_CMD_FILE"

chmod +x "$ROLLBACK_CMD_FILE"

success "Rollback command generated: $ROLLBACK_CMD_FILE"
log ""

# Step 6: Backup current config
log "Step 6: Backing up current configuration..."
CONFIG_FILE="/etc/quantum/ai-engine.env"
if [ -f "$CONFIG_FILE" ]; then
    cp "$CONFIG_FILE" "${CONFIG_FILE}.pre_canary_backup"
    success "Config backed up: ${CONFIG_FILE}.pre_canary_backup"
else
    error "Config file not found: $CONFIG_FILE"
    exit 1
fi
log ""

# Step 7: Set canary weight
log "Step 7: Setting canary model weight..."
if grep -q "^${CANARY_MODEL}_WEIGHT=" "$CONFIG_FILE"; then
    # Update existing weight
    OLD_WEIGHT=$(grep "^${CANARY_MODEL}_WEIGHT=" "$CONFIG_FILE" | cut -d= -f2)
    sed -i "s/^${CANARY_MODEL}_WEIGHT=.*/${CANARY_MODEL}_WEIGHT=$CANARY_WEIGHT/" "$CONFIG_FILE"
    log "Updated: ${CANARY_MODEL}_WEIGHT $OLD_WEIGHT → $CANARY_WEIGHT"
else
    # Add new weight
    echo "${CANARY_MODEL}_WEIGHT=$CANARY_WEIGHT" >> "$CONFIG_FILE"
    log "Added: ${CANARY_MODEL}_WEIGHT=$CANARY_WEIGHT"
fi

# Verify
if grep -q "^${CANARY_MODEL}_WEIGHT=$CANARY_WEIGHT" "$CONFIG_FILE"; then
    success "Weight configured: ${CANARY_MODEL}_WEIGHT=$CANARY_WEIGHT"
else
    error "Failed to set weight"
    exit 1
fi
log ""

# Step 8: Restart AI engine
log "Step 8: Restarting AI engine with canary model..."
systemctl restart quantum-ai-engine.service

# Wait for startup
log "Waiting for AI engine to start..."
sleep 5

if systemctl is-active --quiet quantum-ai-engine.service; then
    success "AI engine running"
else
    error "AI engine failed to start"
    journalctl -u quantum-ai-engine.service --since "$ACTIVATION_TS" --no-pager | tail -50 | tee -a "$ACTIVATION_LOG"
    log ""
    log "ROLLBACK COMMAND:"
    log "  bash $ROLLBACK_CMD_FILE"
    exit 1
fi
log ""

# Step 9: Log activation details
log "Step 9: Recording activation details..."
cat >> "$ACTIVATION_LOG" << EOF

==========================================
CANARY ACTIVATION COMPLETE
==========================================
Start Timestamp: $ACTIVATION_TS
Model ID: $CANARY_MODEL
Weight: $CANARY_WEIGHT (10% traffic)
Cutover: $CUTOVER_TS
Events Analyzed: $EVENT_COUNT
Violations: $VIOLATIONS

Rollback Command:
  bash $ROLLBACK_CMD_FILE

Configuration:
  Config: $CONFIG_FILE
  Backup: ${CONFIG_FILE}.pre_canary_backup
  Log: $ACTIVATION_LOG

Monitoring (6 hours):
  Start: $ACTIVATION_TS
  End: $(date -Iseconds -d "$ACTIVATION_TS + 6 hours")
  
  Commands:
    # Check scoreboard every hour
    make scoreboard
    
    # Check AI engine logs
    journalctl -u quantum-ai-engine.service --since '$ACTIVATION_TS' -f
    
    # Check for violations
    grep -i "violation\|error\|fail" $ACTIVATION_LOG
    
  Violations → IMMEDIATE ROLLBACK:
    bash $ROLLBACK_CMD_FILE

QSC Compliance:
  ✅ NO training (canary only)
  ✅ NO auto-scale (fixed 10%)
  ✅ ONE model (xgb)
  ✅ Quality gate passed
  ✅ Rollback ready

==========================================
EOF

success "Activation details logged"
log ""

# Step 10: Display summary
echo ""
echo "=========================================="
success "CANARY ACTIVATION SUCCESSFUL"
echo "=========================================="
echo ""
echo "Model: $CANARY_MODEL"
echo "Weight: $CANARY_WEIGHT (10% traffic)"
echo "Start: $ACTIVATION_TS"
echo ""
echo "MONITORING (6 hours):"
echo "  - Check scoreboard hourly: make scoreboard"
echo "  - Watch logs: journalctl -u quantum-ai-engine.service -f"
echo "  - End: $(date -Iseconds -d "$ACTIVATION_TS + 6 hours")"
echo ""
echo "IF VIOLATION DETECTED:"
echo "  bash $ROLLBACK_CMD_FILE"
echo ""
echo "LOGS:"
echo "  Activation: $ACTIVATION_LOG"
echo "  Rollback: $ROLLBACK_CMD_FILE"
echo ""
echo "=========================================="
echo ""

exit 0
