#!/bin/bash
# QSC MODE - Quick Start Script
# 
# Usage:
#   ./qsc_quick_start.sh              # Interactive mode
#   ./qsc_quick_start.sh patchtst     # Specific model

set -e

echo ""
echo "================================================================================"
echo "QSC MODE - Quality Safeguard Canary Deployment"
echo "================================================================================"
echo ""

# Default model
MODEL="${1:-patchtst}"

# Check required files
echo "[1/6] Checking prerequisites..."
REQUIRED_FILES=(
    "ops/model_safety/quality_gate.py"
    "ops/model_safety/qsc_mode.py"
    "ops/model_safety/qsc_monitor.py"
    "ops/model_safety/qsc_rollback.sh"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "   ERROR: Missing $file"
        exit 1
    fi
done
echo "   OK: All required files present"
echo ""

# Get cutover timestamp
echo "[2/6] Getting cutover timestamp..."
if command -v systemctl &> /dev/null; then
    CUTOVER=$(systemctl show quantum-ai_engine.service -p ActiveEnterTimestamp | cut -d'=' -f2 | cut -d' ' -f2-4 | xargs -I {} date -d "{}" -u +"%Y-%m-%dT%H:%M:%SZ" 2>/dev/null || echo "2026-01-10T05:43:15Z")
else
    # Fallback for testing
    CUTOVER="2026-01-10T05:43:15Z"
    echo "   WARN: systemctl not available, using test cutover"
fi
echo "   Cutover: $CUTOVER"
echo ""

# Run quality gate (dry run)
echo "[3/6] Running quality gate check..."
if python3 ops/model_safety/qsc_mode.py --model "$MODEL" --cutover "$CUTOVER" --dry-run 2>&1 | tee /tmp/qsc_dryrun.log; then
    echo ""
    echo "   OK: Quality gate passed"
    CAN_ACTIVATE=true
else
    EXIT_CODE=$?
    echo ""
    if [ $EXIT_CODE -eq 1 ]; then
        echo "   BLOCKED: Quality gate failed or insufficient data"
        echo "   Need: exit code 0 + >=200 post-cutover events"
        echo ""
        echo "   Check logs: /tmp/qsc_dryrun.log"
        echo ""
        exit 1
    else
        echo "   ERROR: Activation script error (exit code: $EXIT_CODE)"
        exit 2
    fi
fi
echo ""

# Activate canary
echo "[4/6] Activating canary..."
read -p "   Proceed with canary activation? [y/N] " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "   ABORTED by user"
    exit 0
fi

if python3 ops/model_safety/qsc_mode.py --model "$MODEL" --cutover "$CUTOVER" 2>&1 | tee /tmp/qsc_activate.log; then
    echo ""
    echo "   OK: Canary activated"
else
    echo ""
    echo "   ERROR: Activation failed"
    echo "   Check logs: /tmp/qsc_activate.log"
    exit 2
fi
echo ""

# Restart AI engine
echo "[5/6] Restarting AI engine..."
if command -v systemctl &> /dev/null; then
    if sudo systemctl restart quantum-ai_engine.service 2>/dev/null; then
        echo "   OK: AI engine restarted"
        sleep 2
        sudo systemctl status quantum-ai_engine.service --no-pager -l | head -10
    else
        echo "   WARN: Could not restart (may need: sudo systemctl restart quantum-ai_engine.service)"
    fi
else
    echo "   SKIP: systemctl not available (manual restart required)"
fi
echo ""

# Start monitoring
echo "[6/6] Starting canary monitor..."
echo ""
echo "Options:"
echo "  a) Foreground (see output, Ctrl+C to stop)"
echo "  b) Background (daemon mode)"
echo "  c) Skip (start manually later)"
echo ""
read -p "Choose [a/b/c]: " -n 1 -r
echo ""

case $REPLY in
    a|A)
        echo "Starting monitoring in foreground (6 hours)..."
        echo "Press Ctrl+C to stop early (canary will remain active)"
        echo ""
        python3 ops/model_safety/qsc_monitor.py
        ;;
    b|B)
        echo "Starting monitoring in background..."
        nohup python3 ops/model_safety/qsc_monitor.py > logs/qsc_monitor.log 2>&1 &
        MONITOR_PID=$!
        echo "   PID: $MONITOR_PID"
        echo "   Log: logs/qsc_monitor.log"
        echo "   To stop: kill $MONITOR_PID"
        ;;
    *)
        echo "Skipped monitoring start."
        echo ""
        echo "To start manually:"
        echo "   python3 ops/model_safety/qsc_monitor.py"
        ;;
esac

echo ""
echo "================================================================================"
echo "QSC MODE - Canary Active"
echo "================================================================================"
echo ""
echo "Model:           $MODEL"
echo "Canary Weight:   10%"
echo "Monitor:         6 hours (720 checks @ 30s interval)"
echo "Rollback:        Automatic on violation"
echo ""
echo "Manual rollback: bash ops/model_safety/qsc_rollback.sh"
echo "Status log:      tail -f logs/qsc_canary.jsonl"
echo "Scoreboard:      cat reports/safety/scoreboard_latest.md"
echo ""
echo "================================================================================"
echo ""
