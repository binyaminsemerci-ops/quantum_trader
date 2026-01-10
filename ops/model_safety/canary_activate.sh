#!/bin/bash
# Canary Activation - MANUAL MODEL DEPLOYMENT
# FAIL-CLOSED: Quality gate must pass before activation

set -e

MODEL_NAME="${1}"
MODEL_ENV_KEY="${2}"

if [[ -z "$MODEL_NAME" ]] || [[ -z "$MODEL_ENV_KEY" ]]; then
    echo "Usage: $0 <model_name> <env_key>"
    echo "Example: $0 patchtst PATCHTST_SHADOW_ONLY"
    exit 1
fi

echo "======================================================================"
echo "CANARY ACTIVATION - $MODEL_NAME"
echo "======================================================================"
echo ""

# 1. Run quality gate
echo "[1/5] Running quality gate..."
cd /home/qt/quantum_trader
python3 ops/model_safety/quality_gate.py

if [[ $? -ne 0 ]]; then
    echo ""
    echo "❌ QUALITY GATE FAILED - ABORTING"
    echo "Model activation blocked due to safety violations"
    exit 2
fi

echo "✅ Quality gate passed"
echo ""

# 2. Backup current config
BACKUP_DIR="/opt/quantum/backups/model_activations"
mkdir -p "$BACKUP_DIR"
TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/ai-engine.env.$TIMESTAMP"

echo "[2/5] Backing up config..."
cp /etc/quantum/ai-engine.env "$BACKUP_FILE"
echo "✅ Backup: $BACKUP_FILE"
echo ""

# 3. Get git hash for traceability
GIT_HASH=$(git rev-parse --short HEAD)
echo "[3/5] Git hash: $GIT_HASH"
echo ""

# 4. Update environment (anchored edit)
echo "[4/5] Updating $MODEL_ENV_KEY..."

# Example: PATCHTST_SHADOW_ONLY=false (activate)
# Use sed for anchored edits to avoid full file rewrites
sed -i "s/^${MODEL_ENV_KEY}=.*/${MODEL_ENV_KEY}=false/" /etc/quantum/ai-engine.env

echo "✅ Updated $MODEL_ENV_KEY=false"
echo ""

# 5. Restart service
echo "[5/5] Restarting quantum-ai-engine..."
systemctl restart quantum-ai-engine

sleep 5

# Verify service started
if ! systemctl is-active --quiet quantum-ai-engine; then
    echo ""
    echo "❌ SERVICE FAILED TO START - ROLLING BACK"
    cp "$BACKUP_FILE" /etc/quantum/ai-engine.env
    systemctl restart quantum-ai-engine
    exit 3
fi

echo "✅ Service restarted"
echo ""

# 6. Journal proof (last 20 lines)
echo "======================================================================"
echo "JOURNAL PROOF (last 20 lines)"
echo "======================================================================"
journalctl -u quantum-ai-engine --no-pager -n 20

echo ""
echo "======================================================================"
echo "✅ CANARY ACTIVATION COMPLETE"
echo "======================================================================"
echo ""
echo "Model: $MODEL_NAME"
echo "Env key: $MODEL_ENV_KEY"
echo "Git hash: $GIT_HASH"
echo "Backup: $BACKUP_FILE"
echo ""
echo "NEXT STEPS:"
echo "  1. Monitor trade_intents for $MODEL_NAME predictions"
echo "  2. Check ensemble agreement (make scoreboard)"
echo "  3. If issues: ops/model_safety/rollback_last.sh"
