#!/bin/bash
# Final health check for Quantum Trader V3 auto-repair

echo "=== QUANTUM TRADER V3 - FINAL HEALTH CHECK ==="
echo ""

echo "1. PostgreSQL Database Status:"
docker exec quantum_postgres psql -U quantum -l | grep quantum
echo ""

echo "2. XGBoost Model Features:"
docker exec quantum_ai_engine python3 -c 'import joblib; m=joblib.load("/app/models/xgb_futures_model.joblib"); print(f"✅ Model expects {m.n_features_in_} features")'
echo ""

echo "3. Recent Errors (Last 2 Minutes):"
PG_ERRORS=$(docker logs --since 2m quantum_postgres 2>&1 | grep -ci "fatal")
XGB_ERRORS=$(docker logs --since 2m quantum_ai_engine 2>&1 | grep -ci "mismatch")
GRAFANA_ERRORS=$(docker logs --since 2m quantum_grafana 2>&1 | grep -ci "restart.*plugin")

echo "   PostgreSQL errors: $PG_ERRORS"
echo "   XGBoost errors: $XGB_ERRORS"
echo "   Grafana restart notices: $GRAFANA_ERRORS"
echo ""

TOTAL_ERRORS=$((PG_ERRORS + XGB_ERRORS + GRAFANA_ERRORS))
if [ $TOTAL_ERRORS -eq 0 ]; then
    echo "✅ TOTAL RECENT ERRORS: 0 - System healthy!"
else
    echo "⚠️ TOTAL RECENT ERRORS: $TOTAL_ERRORS"
fi
echo ""

echo "4. Container Status:"
docker ps --filter name=quantum --format "{{.Names}}: {{.Status}}" | head -9
echo ""

echo "=== HEALTH CHECK COMPLETE ==="
