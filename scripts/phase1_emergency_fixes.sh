#!/bin/bash
# ============================================================================
# PHASE 1: EMERGENCY FIXES - Quantum Trader
# ============================================================================
# Deployed: 30. desember 2025
# Duration: 48 timer
# Priority: CRITICAL
# ============================================================================

set -e

echo "🔥 PHASE 1: EMERGENCY FIXES - STARTING"
echo "========================================"
echo ""

# Priority 1.4: Install Missing Dependencies (CatBoost)
echo "📦 Priority 1.4: Installing CatBoost..."
pip install catboost==1.2.2
echo "✅ CatBoost installed"
echo ""

# Verify installation
echo "🔍 Verifying CatBoost..."
python -c "import catboost; print(f'✅ CatBoost version: {catboost.__version__}')"
echo ""

echo "✅ PHASE 1 SCRIPT COMPLETE!"
echo ""
echo "Next steps:"
echo "1. Restart AI Engine: docker restart quantum_ai_engine"
echo "2. Check logs: docker logs quantum_ai_engine --tail 50"
echo "3. Verify Position Monitor: docker ps | grep position_monitor"
echo "4. Test Universe OS: curl http://localhost:8006/health"
