#!/bin/bash
# P3 Switch Apply Layer to TESTNET Mode
# Use with CAUTION - enables real order execution

set -e

echo "=== P3 SWITCH TO TESTNET MODE ==="
echo "⚠️  WARNING: This will enable REAL order execution on Binance testnet"
echo ""
read -p "Are you sure? Type 'YES' to confirm: " CONFIRM

if [ "$CONFIRM" != "YES" ]; then
    echo "❌ Aborted"
    exit 1
fi

echo ""
echo "Step 1: Backing up current config..."
cp /etc/quantum/apply-layer.env /etc/quantum/apply-layer.env.bak.$(date +%s)

echo "Step 2: Setting APPLY_MODE=testnet..."
sed -i 's/^APPLY_MODE=.*/APPLY_MODE=testnet/' /etc/quantum/apply-layer.env

echo "Step 3: Verifying change..."
grep "^APPLY_MODE=" /etc/quantum/apply-layer.env

echo "Step 4: Restarting Apply Layer..."
systemctl restart quantum-apply-layer
sleep 3

echo "Step 5: Checking service status..."
if systemctl is-active --quiet quantum-apply-layer; then
    echo "✅ Apply Layer active in TESTNET mode"
else
    echo "❌ Apply Layer failed to start"
    exit 1
fi

echo ""
echo "=== TESTNET MODE ACTIVE ==="
echo "Governor will enforce limits with REAL Binance data"
echo "Apply Layer will BLOCK if permit missing"
echo ""
echo "Run proof: bash /home/qt/quantum_trader/ops/p3_proof_testnet.sh"
