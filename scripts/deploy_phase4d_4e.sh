#!/bin/bash
# Phase 4D + 4E: Model Supervisor & Predictive Governance Deployment
# Deploy to VPS and verify functionality

set -e

echo "=========================================="
echo "Phase 4D + 4E: Model Supervisor & Governance"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 1. Copy new service to VPS
echo -e "${BLUE}[1/5] Copying model_supervisor_governance.py to VPS...${NC}"
scp -i ~/.ssh/hetzner_fresh \
    backend/microservices/ai_engine/services/model_supervisor_governance.py \
    qt@46.224.116.254:~/quantum_trader/backend/microservices/ai_engine/services/

echo -e "${GREEN}✅ File copied${NC}"
echo ""

# 2. Copy updated service.py to VPS
echo -e "${BLUE}[2/5] Copying updated service.py to VPS...${NC}"
scp -i ~/.ssh/hetzner_fresh \
    microservices/ai_engine/service.py \
    qt@46.224.116.254:~/quantum_trader/microservices/ai_engine/

echo -e "${GREEN}✅ Service.py updated${NC}"
echo ""

# 3. Rebuild AI Engine container
echo -e "${BLUE}[3/5] Rebuilding AI Engine container on VPS...${NC}"
ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 << 'ENDSSH'
cd ~/quantum_trader

# Stop current AI Engine
echo "Stopping AI Engine..."
docker compose stop ai_engine

# Rebuild with no cache
echo "Rebuilding AI Engine (no cache)..."
docker compose build ai_engine --no-cache

# Start AI Engine
echo "Starting AI Engine..."
docker compose up -d ai_engine

# Wait for startup
echo "Waiting for AI Engine to initialize (20 seconds)..."
sleep 20

echo "✅ AI Engine restarted"
ENDSSH

echo -e "${GREEN}✅ Container rebuilt and restarted${NC}"
echo ""

# 4. Verify Supervisor & Governance activation
echo -e "${BLUE}[4/5] Verifying Supervisor & Governance activation...${NC}"
ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 << 'ENDSSH'
echo "Checking logs for Phase 4D+4E activation..."
docker logs quantum_ai_engine --tail 100 | grep -E "Supervisor|Governance|PHASE 4D"

echo ""
echo "Checking for model registration..."
docker logs quantum_ai_engine --tail 100 | grep "Registered"
ENDSSH

echo -e "${GREEN}✅ Verification complete${NC}"
echo ""

# 5. Test health endpoint
echo -e "${BLUE}[5/5] Testing health endpoint for governance status...${NC}"
ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 << 'ENDSSH'
echo "Fetching /health endpoint..."
curl -s http://localhost:8001/health | python3 -m json.tool | grep -A 20 "governance"

echo ""
echo "Testing signal generation endpoint..."
curl -s -X POST http://localhost:8001/api/ai/signal \
     -H "Content-Type: application/json" \
     --data '{"symbol":"BTCUSDT"}' | python3 -m json.tool
ENDSSH

echo ""
echo -e "${GREEN}=========================================="
echo -e "✅ Phase 4D + 4E Deployment Complete!"
echo -e "==========================================${NC}"
echo ""
echo -e "${YELLOW}Expected Results:${NC}"
echo "  ✅ [PHASE 4D+4E] Supervisor + Predictive Governance active"
echo "  ✅ [Supervisor] Registered PatchTST"
echo "  ✅ [Supervisor] Registered NHiTS"
echo "  ✅ [Supervisor] Registered XGBoost"
echo "  ✅ [Supervisor] Registered LightGBM"
echo "  ✅ [Governance] Adjusted weights"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "  1. Monitor logs: ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 'docker logs -f quantum_ai_engine'"
echo "  2. Check governance status: curl http://46.224.116.254:8001/health | jq '.metrics.governance'"
echo "  3. Test signal generation to trigger governance cycles"
echo ""
