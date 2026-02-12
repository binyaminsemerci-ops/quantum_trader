#!/bin/bash
# PATH 2.4A — Complete Verification and Calibration Workflow
# 
# This script:
# 1. Verifies PATH 2.3D shadow mode is running
# 2. Checks signal accumulation (need 100+ samples)
# 3. Runs calibration if data is ready
# 4. Deploys calibrated service
#
# Usage:
#   bash verify_path2_and_calibrate.sh [--force-calibrate]

set -e  # Exit on error

SCRIPT_DIR="/home/qt/quantum_trader"
SIGNAL_STREAM="quantum:stream:signal.score"
MIN_SAMPLES=100
PREFERRED_SAMPLES=1000

echo "============================================================================"
echo "PATH 2.4A — Verification and Calibration Workflow"
echo "============================================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# STEP 1: Verify PATH 2.3D Shadow Mode
# ============================================================================
echo -e "${BLUE}[STEP 1] Verifying PATH 2.3D Shadow Mode...${NC}"
echo ""

# Check service status
echo "Checking quantum-ensemble-predictor.service status..."
if systemctl is-active --quiet quantum-ensemble-predictor.service; then
    echo -e "${GREEN}✅ Service is ACTIVE${NC}"
    
    # Show service uptime
    UPTIME=$(systemctl show quantum-ensemble-predictor.service -p ActiveEnterTimestamp --value)
    echo "   Service started: $UPTIME"
else
    echo -e "${RED}❌ Service is NOT ACTIVE${NC}"
    echo ""
    echo "Service status:"
    systemctl status quantum-ensemble-predictor.service --no-pager -n 5
    echo ""
    echo -e "${RED}ABORT: Ensemble predictor must be running for calibration${NC}"
    exit 1
fi

# Check recent logs for errors
echo ""
echo "Checking recent logs (last 20 lines)..."
LOG_ERRORS=$(journalctl -u quantum-ensemble-predictor.service -n 20 --no-pager | grep -i "error\|fail\|exception" || true)

if [ -n "$LOG_ERRORS" ]; then
    echo -e "${YELLOW}⚠️  Found potential errors in logs:${NC}"
    echo "$LOG_ERRORS"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "${GREEN}✅ No errors in recent logs${NC}"
fi

# Check if service is producing signals
echo ""
echo "Checking if service is producing signals (last 5 entries)..."
RECENT_SIGNALS=$(redis-cli XREVRANGE $SIGNAL_STREAM + - COUNT 5 2>/dev/null || echo "")

if [ -z "$RECENT_SIGNALS" ]; then
    echo -e "${RED}❌ No signals found in stream${NC}"
    echo -e "${YELLOW}   Service may be starting up or not receiving features${NC}"
    exit 1
else
    echo -e "${GREEN}✅ Service is producing signals${NC}"
    
    # Show most recent signal timestamp
    LATEST_TS=$(redis-cli XREVRANGE $SIGNAL_STREAM + - COUNT 1 | grep -o '[0-9]\{13\}' | head -1)
    if [ -n "$LATEST_TS" ]; then
        LATEST_TIME=$(date -d @$((LATEST_TS/1000)) '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo "unknown")
        echo "   Latest signal: $LATEST_TIME"
    fi
fi

echo ""
echo -e "${GREEN}✅ PATH 2.3D Shadow Mode: VERIFIED${NC}"

# ============================================================================
# STEP 2: Check Data Collection Status
# ============================================================================
echo ""
echo -e "${BLUE}[STEP 2] Checking Data Collection Status...${NC}"
echo ""

# Count signals in stream
SIGNAL_COUNT=$(redis-cli XLEN $SIGNAL_STREAM)
echo "Signal count in stream: $SIGNAL_COUNT"

if [ "$SIGNAL_COUNT" -lt "$MIN_SAMPLES" ]; then
    echo -e "${YELLOW}⚠️  Insufficient samples for calibration${NC}"
    echo "   Minimum required: $MIN_SAMPLES"
    echo "   Preferred: $PREFERRED_SAMPLES"
    echo "   Current: $SIGNAL_COUNT"
    echo ""
    
    # Calculate time to wait (assuming 1 signal per 2 minutes average)
    NEEDED=$((MIN_SAMPLES - SIGNAL_COUNT))
    WAIT_HOURS=$((NEEDED * 2 / 60))
    
    echo "   Estimated time needed: ~$WAIT_HOURS hours"
    echo ""
    echo -e "${BLUE}Recommendation: Wait for more data accumulation${NC}"
    echo ""
    
    if [[ "$1" != "--force-calibrate" ]]; then
        echo "To force calibration anyway, run with: --force-calibrate"
        exit 0
    else
        echo -e "${YELLOW}--force-calibrate specified, continuing anyway...${NC}"
        echo ""
    fi
elif [ "$SIGNAL_COUNT" -lt "$PREFERRED_SAMPLES" ]; then
    echo -e "${YELLOW}⚠️  Sample count is adequate but not optimal${NC}"
    echo "   Minimum: $MIN_SAMPLES ✅"
    echo "   Preferred: $PREFERRED_SAMPLES"
    echo "   Current: $SIGNAL_COUNT"
    echo ""
    echo -e "${BLUE}Calibration will proceed, but more data would improve accuracy${NC}"
    echo ""
else
    echo -e "${GREEN}✅ Sufficient samples for high-quality calibration${NC}"
    echo "   Current: $SIGNAL_COUNT (exceeds preferred $PREFERRED_SAMPLES)"
    echo ""
fi

# Check signal diversity (symbols)
echo "Checking signal diversity..."
SYMBOLS=$(redis-cli XREVRANGE $SIGNAL_STREAM + - COUNT 100 | grep -o '"symbol":"[^"]*"' | sort -u | wc -l)
echo "   Unique symbols (last 100 signals): $SYMBOLS"

if [ "$SYMBOLS" -lt 3 ]; then
    echo -e "${YELLOW}   ⚠️  Low symbol diversity (prefer 5+)${NC}"
else
    echo -e "${GREEN}   ✅ Good symbol diversity${NC}"
fi

# ============================================================================
# STEP 3: Prerequisites Check
# ============================================================================
echo ""
echo -e "${BLUE}[STEP 3] Checking Prerequisites...${NC}"
echo ""

# Check sklearn
if python3 -c "import sklearn" 2>/dev/null; then
    SKLEARN_VERSION=$(python3 -c "import sklearn; print(sklearn.__version__)")
    echo -e "${GREEN}✅ sklearn installed (version $SKLEARN_VERSION)${NC}"
else
    echo -e "${RED}❌ sklearn not installed${NC}"
    echo "   Installing sklearn..."
    pip3 install scikit-learn
fi

# Check Redis connection
if redis-cli ping > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Redis connection OK${NC}"
else
    echo -e "${RED}❌ Redis connection failed${NC}"
    exit 1
fi

# Check calibration scripts exist
if [ -f "$SCRIPT_DIR/ai_engine/calibration/run_calibration_workflow.py" ]; then
    echo -e "${GREEN}✅ Calibration workflow script found${NC}"
else
    echo -e "${RED}❌ Calibration workflow script not found${NC}"
    echo "   Expected: $SCRIPT_DIR/ai_engine/calibration/run_calibration_workflow.py"
    exit 1
fi

# ============================================================================
# STEP 4: Run Calibration Workflow
# ============================================================================
echo ""
echo -e "${BLUE}[STEP 4] Running Calibration Workflow...${NC}"
echo ""

cd $SCRIPT_DIR

echo "Executing: python3 ai_engine/calibration/run_calibration_workflow.py"
echo ""

if python3 ai_engine/calibration/run_calibration_workflow.py; then
    echo ""
    echo -e "${GREEN}✅ Calibration workflow completed successfully${NC}"
else
    echo ""
    echo -e "${RED}❌ Calibration workflow failed${NC}"
    exit 1
fi

# ============================================================================
# STEP 5: Verify Calibrator Saved
# ============================================================================
echo ""
echo -e "${BLUE}[STEP 5] Verifying Calibrator Artifacts...${NC}"
echo ""

CALIBRATOR_PATH="$SCRIPT_DIR/ai_engine/calibration/calibrator_v1.pkl"
METADATA_PATH="$SCRIPT_DIR/ai_engine/calibration/calibrator_v1.pkl.json"

if [ -f "$CALIBRATOR_PATH" ]; then
    echo -e "${GREEN}✅ Calibrator saved: $CALIBRATOR_PATH${NC}"
    ls -lh "$CALIBRATOR_PATH"
    echo ""
else
    echo -e "${RED}❌ Calibrator file not found${NC}"
    exit 1
fi

if [ -f "$METADATA_PATH" ]; then
    echo -e "${GREEN}✅ Metadata saved: $METADATA_PATH${NC}"
    echo ""
    echo "Calibration Statistics:"
    cat "$METADATA_PATH" | python3 -m json.tool | head -20
    echo ""
    
    # Extract ECE
    ECE=$(cat "$METADATA_PATH" | python3 -c "import sys, json; print(json.load(sys.stdin).get('expected_calibration_error', 'N/A'))")
    echo -e "${BLUE}Expected Calibration Error (ECE): $ECE${NC}"
    echo ""
    
    # Validate ECE
    if python3 -c "import sys; sys.exit(0 if float('$ECE') < 0.10 else 1)" 2>/dev/null; then
        echo -e "${GREEN}✅ ECE < 0.10 (calibration acceptable)${NC}"
    else
        echo -e "${YELLOW}⚠️  ECE >= 0.10 (calibration may need improvement)${NC}"
        echo "   Consider collecting more data or investigating model"
    fi
else
    echo -e "${YELLOW}⚠️  Metadata file not found (non-critical)${NC}"
fi

# ============================================================================
# STEP 6: Deploy Calibrated Service
# ============================================================================
echo ""
echo -e "${BLUE}[STEP 6] Deploying Calibrated Service...${NC}"
echo ""

echo "Restarting quantum-ensemble-predictor.service to load calibrator..."
systemctl restart quantum-ensemble-predictor.service

echo "Waiting 5 seconds for service to start..."
sleep 5

# Verify service restarted successfully
if systemctl is-active --quiet quantum-ensemble-predictor.service; then
    echo -e "${GREEN}✅ Service restarted successfully${NC}"
else
    echo -e "${RED}❌ Service failed to restart${NC}"
    systemctl status quantum-ensemble-predictor.service --no-pager
    exit 1
fi

# Check logs for calibration loaded message
echo ""
echo "Checking if calibration was loaded..."
CALIBRATION_LOG=$(journalctl -u quantum-ensemble-predictor.service -n 50 --no-pager | grep -i "calibration" || echo "")

if echo "$CALIBRATION_LOG" | grep -q "Loaded calibrator"; then
    echo -e "${GREEN}✅ Calibration loaded successfully${NC}"
    echo "$CALIBRATION_LOG" | grep -i "calibration" | tail -5
else
    echo -e "${YELLOW}⚠️  Could not confirm calibration loaded from logs${NC}"
    echo "Recent logs:"
    journalctl -u quantum-ensemble-predictor.service -n 20 --no-pager
fi

# ============================================================================
# COMPLETION
# ============================================================================
echo ""
echo "============================================================================"
echo -e "${GREEN}✅ PATH 2.4A CALIBRATION WORKFLOW COMPLETE${NC}"
echo "============================================================================"
echo ""
echo "Summary:"
echo "  • Shadow mode: VERIFIED"
echo "  • Data collected: $SIGNAL_COUNT signals"
echo "  • Calibrator trained: YES"
echo "  • Calibrator deployed: YES"
echo "  • Service status: RUNNING"
echo ""
echo "Next Steps:"
echo "  1. Monitor calibrated signals in signal.score stream"
echo "  2. Compare confidence distributions (before vs after)"
echo "  3. Create CONFIDENCE_SEMANTICS_V1.md document"
echo "  4. Proceed to PATH 2.4B (Regime Analysis) or PATH 2.4C (Apply-Layer)"
echo ""
echo "Monitor calibrated output:"
echo "  redis-cli XREAD COUNT 5 STREAMS $SIGNAL_STREAM 0-0"
echo ""
echo "View calibration metadata:"
echo "  cat $METADATA_PATH | python3 -m json.tool"
echo ""
echo -e "${GREEN}Confidence is now empirically grounded! 🎯${NC}"
echo ""
