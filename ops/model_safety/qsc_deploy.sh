#!/bin/bash
# QSC FAIL-CLOSED DEPLOYMENT - IMMEDIATE ACTIONS
# Run these commands on VPS (46.224.116.254)
# Date: 2026-01-11
# Commit: d13e87bd

set -e

echo "============================================"
echo "QSC FAIL-CLOSED DEPLOYMENT - VPS COMMANDS"
echo "============================================"
echo ""

# ============================================
# STEP 1: Deploy Code
# ============================================
echo "[STEP 1] Deploying code..."
cd /home/qt/quantum_trader
git fetch origin
git checkout main
git pull origin main

# Verify commit
COMMIT=$(git log --oneline -1 | awk '{print $1}')
echo "✅ Deployed commit: $COMMIT"
echo ""

# Restart service (stop + start for clean reload)
echo "Stopping service..."
sudo systemctl stop quantum-ai-engine.service
sleep 2

echo "Starting service..."
sudo systemctl start quantum-ai-engine.service
echo "✅ Service restarted (stop + start)"
echo ""

# Wait for startup
sleep 5

# Get cutover timestamp
CUTOVER=$(systemctl show -p ActiveEnterTimestamp quantum-ai-engine.service | awk '{print $2" "$3}')
CUTOVER_ISO=$(date -d "$CUTOVER" -u +%Y-%m-%dT%H:%M:%SZ)
export CUTOVER_ISO
echo "📅 Cutover timestamp: $CUTOVER_ISO"
echo ""

# Check startup logs
echo "[STEP 1] Checking startup logs..."
journalctl -u quantum-ai-engine.service --since "30 seconds ago" | tail -30

echo ""
echo "Verifying AI_ENGINE_BUILD version..."
BUILD_SHA=$(journalctl -u quantum-ai-engine.service --since "30 seconds ago" | grep "AI_ENGINE_BUILD:" | tail -1 | awk '{print $NF}')
if [ -n "$BUILD_SHA" ]; then
    echo "✅ AI_ENGINE_BUILD: $BUILD_SHA (expected: $COMMIT)"
    if [ "$BUILD_SHA" != "$COMMIT" ]; then
        echo "⚠️  WARNING: Build SHA mismatch! Service may be using cached code."
        echo "   Try: sudo systemctl stop && sleep 5 && sudo systemctl start"
    fi
else
    echo "⚠️  WARNING: AI_ENGINE_BUILD not found in logs (version check failed)"
fi
echo ""

# ============================================
# STEP 2: Wait for Data (2-3 min)
# ============================================
echo "[STEP 2] Waiting 180 seconds for data collection..."
sleep 180
echo "✅ Wait complete"
echo ""

# ============================================
# STEP 3: Feature Sanity Check
# ============================================
echo "[STEP 3] Running feature sanity check..."
/opt/quantum/venvs/ai-engine/bin/python ops/model_safety/feature_sanity.py \
  --after "$CUTOVER_ISO" \
  --count 200

SANITY_RC=$?
echo ""
echo "Feature Sanity Exit Code: $SANITY_RC"
echo ""

if [ $SANITY_RC -ne 0 ]; then
    echo "❌ BLOCKED: Feature sanity failed (RC=$SANITY_RC)"
    echo "   Fix feature pipeline before proceeding"
    exit 1
fi

echo "✅ Feature sanity PASSED"
echo ""

# ============================================
# STEP 4: Quality Gate (Collection Mode)
# ============================================
echo "[STEP 4] Running quality gate (collection mode)..."
/opt/quantum/venvs/ai-engine/bin/python ops/model_safety/quality_gate.py \
  --mode collection \
  --after "$CUTOVER_ISO"

COLLECTION_RC=$?
echo ""
echo "Collection Mode Exit Code: $COLLECTION_RC"
echo ""

if [ $COLLECTION_RC -eq 3 ]; then
    echo "✅ Collection complete (RC=3 is expected)"
elif [ $COLLECTION_RC -eq 2 ]; then
    echo "⚠️  BLOCKERS detected in collection mode"
    echo "   Check report: reports/safety/quality_gate_*_collection_post_cutover.md"
    echo "   Models may need retraining"
    echo ""
    echo "Manual next steps:"
    echo "1. Read collection report for model analysis"
    echo "2. If insufficient data: wait longer, rerun"
    echo "3. If quality violations: diagnose model collapse (drift, features)"
    echo "4. Do NOT proceed to canary mode until blockers resolved"
    exit 2
else
    echo "❌ UNEXPECTED: Collection mode should exit 3, got $COLLECTION_RC"
    exit 1
fi

echo ""

# ============================================
# STEP 5: Count Events (Check for >=200)
# ============================================
echo "[STEP 5] Counting post-cutover events..."
EVENT_COUNT=$(python3 <<PY
import subprocess
def sh(c): return subprocess.check_output(c,shell=True,text=True).strip()
count=0; cur="$(date -d "$CUTOVER" +%s)000-0"
for _ in range(30):
  out=sh(f"redis-cli --raw XRANGE quantum:stream:trade.intent {cur} + COUNT 200")
  if not out: break
  ids=[ln for ln in out.split("\n") if "-" in ln and ln.replace("-","").isdigit()]
  if not ids: break
  if cur in ids: ids.remove(cur)
  count+=len(ids); cur=ids[-1] if ids else cur
print(count)
PY
)

echo "📊 Post-cutover events: $EVENT_COUNT"
echo ""

if [ "$EVENT_COUNT" -lt 200 ]; then
    echo "⚠️  INSUFFICIENT DATA: Need >=200 events for canary mode, have $EVENT_COUNT"
    echo "   Wait time estimate: $((5 * (200 - EVENT_COUNT) / 60)) minutes (at ~60 events/min)"
    echo ""
    echo "Run this to wait and recount:"
    echo "  sleep 120 && bash $(basename $0) --step canary"
    exit 0
fi

echo "✅ Sufficient data for canary mode ($EVENT_COUNT >= 200)"
echo ""

# ============================================
# STEP 6: Quality Gate (Canary Mode)
# ============================================
echo "[STEP 6] Running quality gate (CANARY MODE)..."
/opt/quantum/venvs/ai-engine/bin/python ops/model_safety/quality_gate.py \
  --mode canary \
  --after "$CUTOVER_ISO"

CANARY_RC=$?
echo ""
echo "Canary Mode Exit Code: $CANARY_RC"
echo ""

if [ $CANARY_RC -eq 0 ]; then
    echo "✅ QUALITY GATE PASSED (RC=0)"
    echo ""
    echo "============================================"
    echo "🚀 READY FOR CANARY ACTIVATION"
    echo "============================================"
    echo ""
    echo "Read report: reports/safety/quality_gate_*_post_cutover.md"
    echo ""
    echo "Identify which model PASSED (look for ✅ PASS in report)"
    echo ""
    echo "Manual next step:"
    echo "  bash ops/model_safety/qsc_activate_canary.sh <model_name> \"$CUTOVER_ISO\""
    echo ""
    echo "Example:"
    echo "  bash ops/model_safety/qsc_activate_canary.sh patchtst \"$CUTOVER_ISO\""
    echo ""
elif [ $CANARY_RC -eq 2 ]; then
    echo "❌ BLOCKED: Quality gate failed (RC=2)"
    echo "   Check report: reports/safety/quality_gate_*_post_cutover.md"
    echo ""
    echo "Common causes:"
    echo "- Model collapse (constant output, PSI=1.000 drift)"
    echo "- All models showing >95% same action"
    echo "- No models passed diversity checks"
    echo ""
    echo "Next steps:"
    echo "1. Read report for specific violations"
    echo "2. Check model logs: journalctl -u quantum-ai-engine.service | grep 'QSC FAIL-CLOSED'"
    echo "3. Retrain models on recent data to reduce drift"
    echo "4. DO NOT lower quality gate thresholds"
    exit 2
else
    echo "❌ UNEXPECTED: Canary mode exit code $CANARY_RC"
    exit 1
fi

echo ""
echo "============================================"
echo "DEPLOYMENT COMPLETE"
echo "============================================"
