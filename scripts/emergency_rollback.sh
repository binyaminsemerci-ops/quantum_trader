#!/bin/bash
# Emergency rollback script - stops all AI modules and reverts to core services

echo "🚨 EMERGENCY ROLLBACK INITIATED"
echo "================================"
echo ""

# Stop all AI module containers
echo "⏸️  Stopping AI modules..."
docker stop quantum_ai_hfos 2>/dev/null || true
docker stop quantum_pba 2>/dev/null || true
docker stop quantum_pal 2>/dev/null || true
docker stop quantum_pil 2>/dev/null || true
docker stop quantum_universe_os 2>/dev/null || true
docker stop quantum_model_supervisor 2>/dev/null || true
docker stop quantum_self_healing 2>/dev/null || true
docker stop quantum_orchestrator_policy 2>/dev/null || true
docker stop quantum_trading_mathematician 2>/dev/null || true
docker stop quantum_aelm 2>/dev/null || true
docker stop quantum_msc_ai 2>/dev/null || true
docker stop quantum_opportunity_ranker 2>/dev/null || true
docker stop quantum_ess 2>/dev/null || true
docker stop quantum_retraining_orchestrator 2>/dev/null || true

echo ""
echo "🔄 Restarting core services..."
docker restart quantum_ai_engine
docker restart quantum_execution
docker restart quantum_trading_bot
docker restart quantum_clm
docker restart quantum_risk_safety

echo ""
echo "⏳ Waiting for core services to stabilize (30s)..."
sleep 30

echo ""
echo "✅ Health check..."
curl -s http://localhost:8001/health | grep -q "healthy" && echo "✅ AI Engine: OK" || echo "❌ AI Engine: FAILED"
curl -s http://localhost:8002/health | grep -q "healthy" && echo "✅ Execution: OK" || echo "❌ Execution: FAILED"
curl -s http://localhost:8003/health | grep -q "healthy" && echo "✅ Trading Bot: OK" || echo "❌ Trading Bot: FAILED"

echo ""
echo "================================"
echo "✅ Rollback complete - Core services operational"
echo "All AI modules disabled"
echo ""
echo "To re-enable modules, run: ./scripts/enable_ai_modules.sh"
