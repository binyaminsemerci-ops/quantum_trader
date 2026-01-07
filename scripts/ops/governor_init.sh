#!/bin/bash
# Initialize governor with SAFE defaults
# Run ONCE before starting quantum-exec.target

set -e

echo "🛡️  Initializing Quantum Governor..."

# Safe defaults - KILL mode
redis-cli SET quantum:kill 1
echo "✓ quantum:kill = 1 (KILL MODE - execution blocked)"

# TESTNET mode
redis-cli SET quantum:mode TESTNET
echo "✓ quantum:mode = TESTNET"

# Enable governor protection
redis-cli SET quantum:governor:execution ENABLED
echo "✓ quantum:governor:execution = ENABLED"

# Verify
echo ""
echo "Current Governor State:"
redis-cli MGET quantum:kill quantum:mode quantum:governor:execution

echo ""
echo "🛡️  Governor initialized with SAFE defaults"
echo ""
echo "⚠️  IMPORTANT:"
echo "   - Execution is BLOCKED (kill=1)"
echo "   - Start services and verify signals flow"
echo "   - Check logs for '🛑 BLOCKED' messages"
echo "   - Only set kill=0 after verification"
echo ""
echo "To enable trading (DANGEROUS):"
echo "  redis-cli SET quantum:kill 0"
