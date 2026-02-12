#!/bin/bash
# Quick check for ensemble predictor shadow mode

echo "=================================================="
echo "ENSEMBLE PREDICTOR - SHADOW MODE STATUS"
echo "PATH 2.3D | $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================="

# Service status
echo -e "\n🔧 SERVICE STATUS:"
systemctl is-active quantum-ensemble-predictor.service && echo "   ✅ Active" || echo "   ❌ Inactive"

# Recent logs
echo -e "\n📋 RECENT LOGS (last 10 lines):"
journalctl -u quantum-ensemble-predictor.service -n 10 --no-pager | tail -10

# Stream check
echo -e "\n📡 STREAM: quantum:stream:signal.score"
redis-cli EXISTS quantum:stream:signal.score && {
    LENGTH=$(redis-cli XLEN quantum:stream:signal.score)
    echo "   Length: $LENGTH messages"
    
    if [ "$LENGTH" -gt 0 ]; then
        echo -e "\n   Last 3 signals:"
        redis-cli XREVRANGE quantum:stream:signal.score + - COUNT 3 | head -20
    fi
} || echo "   ⚠️  Stream does not exist yet"

# Consumer groups
echo -e "\n👥 CONSUMER GROUPS:"
redis-cli XINFO GROUPS quantum:stream:signal.score 2>/dev/null || echo "   None (shadow mode - no consumers)"

echo -e "\n=================================================="
echo "🔍 SHADOW MODE: Observation only, NO consumption"
echo "📋 Authority: SCORER ONLY"
echo "=================================================="
