#!/bin/bash

echo "=== QUANTUM TRADER CONFIDENCE STATUS ==="
echo ""

ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 << 'ENDSSH'

echo "1. CURRENT CONSENSUS SIGNAL:"
docker exec quantum_redis redis-cli GET quantum:federation:consensus | jq '.'
echo ""

echo "2. FEDERATION METRICS:"
docker exec quantum_redis redis-cli GET quantum:federation:metrics | jq '.'
echo ""

echo "3. TRUST WEIGHTS (All Models):"
echo "  XGBoost:"
docker exec quantum_redis redis-cli GET quantum:trust:xgb
echo "  LightGBM:"
docker exec quantum_redis redis-cli GET quantum:trust:lgbm
echo "  PatchTST:"
docker exec quantum_redis redis-cli GET quantum:trust:patchtst
echo "  N-HiTS:"
docker exec quantum_redis redis-cli GET quantum:trust:nhits
echo "  RL Sizer:"
docker exec quantum_redis redis-cli GET quantum:trust:rl_sizer
echo "  Evo Model:"
docker exec quantum_redis redis-cli GET quantum:trust:evo_model
echo ""

echo "4. MODEL SIGNALS:"
echo "  XGBoost Signal:"
docker exec quantum_redis redis-cli GET quantum:model:xgb:signal
echo "  LightGBM Signal:"
docker exec quantum_redis redis-cli GET quantum:model:lgbm:signal
echo "  PatchTST Signal:"
docker exec quantum_redis redis-cli GET quantum:model:patchtst:signal
echo ""

echo "5. EXECUTION METRICS:"
docker exec quantum_redis redis-cli GET execution_metrics | jq '.'
echo ""

echo "6. LATEST SYSTEM METRICS:"
docker exec quantum_redis redis-cli GET latest_metrics | jq '.'
echo ""

echo "7. RL REWARD HISTORY (Last 10):"
docker exec quantum_redis redis-cli LRANGE rl_reward_history 0 9
echo ""

echo "8. REGIME FORECAST:"
docker exec quantum_redis redis-cli GET latest_regime_forecast
echo ""

ENDSSH
