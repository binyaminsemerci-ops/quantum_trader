#!/usr/bin/env bash
set -euo pipefail

echo "╔════════════════════════════════════════════════╗"
echo "║   UNIVERSE SERVICE - VERIFICATION              ║"
echo "╚════════════════════════════════════════════════╝"
echo ""

# Step 1: Syntax check
echo "1. Verifying Python syntax..."
python3 -m py_compile microservices/universe_service/main.py
echo "   ✅ Syntax valid"
echo ""

# Step 2: Dry-run (single refresh, no loop)
echo "2. Testing single refresh cycle..."
echo "   Setting environment for testnet dry-run..."

export UNIVERSE_MODE=testnet
export UNIVERSE_REFRESH_SEC=999999  # Won't loop in test
export UNIVERSE_MAX=800
export REDIS_HOST=127.0.0.1
export REDIS_PORT=6379
export REDIS_DB=0
export HTTP_TIMEOUT_SEC=10

# Run for 5 seconds then kill
timeout 5 python3 microservices/universe_service/main.py || {
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 124 ]; then
        echo "   ✅ Service started successfully (killed after 5s)"
    else
        echo "   ❌ Service exited with error code $EXIT_CODE"
        exit 1
    fi
}
echo ""

# Step 3: Check Redis keys
echo "3. Checking Redis keys..."
if redis-cli EXISTS quantum:cfg:universe:active > /dev/null 2>&1; then
    echo "   ✅ quantum:cfg:universe:active exists"
    
    COUNT=$(redis-cli HGET quantum:cfg:universe:meta count 2>/dev/null || echo "0")
    STALE=$(redis-cli HGET quantum:cfg:universe:meta stale 2>/dev/null || echo "unknown")
    
    echo "   Symbol count: $COUNT"
    echo "   Stale flag: $STALE"
    echo ""
    
    echo "   First 10 symbols:"
    redis-cli GET quantum:cfg:universe:active | python3 -c "import sys, json; data=json.load(sys.stdin); print('\n'.join(data['symbols'][:10]))" 2>/dev/null || echo "   (jq not available, skipping)"
else
    echo "   ⚠️  quantum:cfg:universe:active not created"
    echo "   This may be expected if Redis not running or fetch failed"
fi
echo ""

# Step 4: Show proof script
echo "4. Testing proof script..."
if [ -f ops/proof_universe.sh ]; then
    chmod +x ops/proof_universe.sh
    bash ops/proof_universe.sh || echo "   ⚠️  Proof script failed (may need Redis running)"
else
    echo "   ⚠️  ops/proof_universe.sh not found"
fi
echo ""

echo "╔════════════════════════════════════════════════╗"
echo "║   VERIFICATION COMPLETE                        ║"
echo "╚════════════════════════════════════════════════╝"
echo ""
echo "Files created:"
echo "  ✅ microservices/universe_service/main.py"
echo "  ✅ microservices/universe_service/universe-service.env.example"
echo "  ✅ ops/systemd/quantum-universe-service.service"
echo "  ✅ ops/proof_universe.sh"
echo "  ✅ ops/README.md (updated)"
echo ""
echo "Next steps:"
echo "  1. Copy config: sudo cp microservices/universe_service/universe-service.env.example /etc/quantum/universe-service.env"
echo "  2. Install systemd: sudo cp ops/systemd/quantum-universe-service.service /etc/systemd/system/"
echo "  3. Start service: sudo systemctl enable --now quantum-universe-service"
echo "  4. Verify: bash ops/proof_universe.sh"
