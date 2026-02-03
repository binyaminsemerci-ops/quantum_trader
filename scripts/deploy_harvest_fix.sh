#!/usr/bin/env bash
# Deploy harvest system fix to VPS
# TASK: Implement CLOSE handler + fix harvest-brain offset

set -euo pipefail

VPS_HOST="${VPS_HOST:-root@46.224.116.254}"
VPS_SSH_KEY="${VPS_SSH_KEY:-~/.ssh/hetzner_fresh}"
REPO_PATH="/home/qt/quantum_trader"

echo "=================================================="
echo "    HARVEST SYSTEM FIX DEPLOYMENT"
echo "=================================================="
echo "VPS: $VPS_HOST"
echo "Repo: $REPO_PATH"
echo

# Step 1: Git pull latest code
echo "=== Step 1: Pull latest code ==="
ssh -i "$VPS_SSH_KEY" "$VPS_HOST" "cd $REPO_PATH && git pull"
echo "✅ Code pulled"
echo

# Step 2: Stop harvest-brain (currently stuck on old events)
echo "=== Step 2: Stop harvest-brain (stuck on backlog) ==="
ssh -i "$VPS_SSH_KEY" "$VPS_HOST" "systemctl stop quantum-harvest-brain"
echo "✅ harvest-brain stopped"
echo

# Step 3: Fix harvest-brain consumer group offset
echo "=== Step 3: Fix harvest-brain offset (skip old events) ==="
ssh -i "$VPS_SSH_KEY" "$VPS_HOST" "cd $REPO_PATH && bash scripts/fix_harvest_brain_offset.sh"
echo "✅ Offset fixed to $ (new messages only)"
echo

# Step 4: Restart apply-layer (with CLOSE handler)
echo "=== Step 4: Restart apply-layer (CLOSE handler active) ==="
ssh -i "$VPS_SSH_KEY" "$VPS_HOST" "systemctl restart quantum-apply-layer"
sleep 3
ssh -i "$VPS_SSH_KEY" "$VPS_HOST" "systemctl status quantum-apply-layer --no-pager | head -15"
echo "✅ apply-layer restarted with CLOSE handler"
echo

# Step 5: Restart harvest-brain (with fixed offset)
echo "=== Step 5: Restart harvest-brain (clean slate) ==="
ssh -i "$VPS_SSH_KEY" "$VPS_HOST" "systemctl start quantum-harvest-brain"
sleep 2
ssh -i "$VPS_SSH_KEY" "$VPS_HOST" "systemctl status quantum-harvest-brain --no-pager | head -15"
echo "✅ harvest-brain restarted"
echo

# Step 6: Verify services
echo "=== Step 6: Verify services ==="
echo "Checking apply-layer logs for CLOSE activity..."
ssh -i "$VPS_SSH_KEY" "$VPS_HOST" "journalctl -u quantum-apply-layer --since '30 seconds ago' | grep '\[CLOSE\]' | head -10 || echo 'No CLOSE activity yet (waiting for exitbrain proposals)'"
echo

echo "Checking harvest-brain logs for new activity..."
ssh -i "$VPS_SSH_KEY" "$VPS_HOST" "tail -20 /mnt/HC_Volume_104287969/quantum-logs/harvest_brain.log || journalctl -u quantum-harvest-brain --since '30 seconds ago' | tail -10"
echo

echo "=================================================="
echo "    DEPLOYMENT COMPLETE"
echo "=================================================="
echo
echo "Next steps:"
echo "  1. Run verification: bash scripts/verify_harvest_system.sh"
echo "  2. Monitor for CLOSE executions (may take minutes for exitbrain to generate new proposals)"
echo "  3. Check apply.result for executed=True + reduceOnly=True"
echo
echo "VPS proof commands (run on VPS):"
echo "  # Check CLOSE plans"
echo "  redis-cli XREVRANGE quantum:stream:apply.plan + - COUNT 100 | grep -B5 FULL_CLOSE_PROPOSED | head -60"
echo
echo "  # Check CLOSE executions"
echo "  redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 50 | grep -B10 'reduceOnly' | head -100"
echo
echo "  # Check apply-layer CLOSE logs"
echo "  journalctl -u quantum-apply-layer --since '5 minutes ago' | grep CLOSE"
echo
echo "  # Check position changes"
echo "  redis-cli KEYS 'quantum:position:*' | xargs -I {} redis-cli HGETALL {}"
