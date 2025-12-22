#!/bin/bash
# ============================================================================
# PHASE 2: DEPLOY CIRCUIT BREAKER API + REDIS MANAGER TIL VPS
# ============================================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

VPS_IP="46.224.116.254"
VPS_USER="qt"
SSH_KEY="$HOME/.ssh/hetzner_fresh"

echo -e "${BLUE}════════════════════════════════════════════════${NC}"
echo -e "${BLUE}⚡ PHASE 2: CIRCUIT BREAKER + REDIS FIX${NC}"
echo -e "${BLUE}════════════════════════════════════════════════${NC}"
echo ""

# STEG 1: GIT PULL
echo -e "${YELLOW}1️⃣ Git pull...${NC}"
ssh -i $SSH_KEY $VPS_USER@$VPS_IP << 'ENDSSH'
cd ~/quantum_trader
git config --global --add safe.directory ~/quantum_trader
git pull origin main 2>&1 | grep -E "(Already|Updating|Fast-forward|error)" || true
ENDSSH
echo -e "${GREEN}✅ Git pull attempt done${NC}"

# STEG 2: REBUILD CONTAINERS
echo ""
echo -e "${YELLOW}2️⃣ Rebuilding containers...${NC}"
ssh -i $SSH_KEY $VPS_USER@$VPS_IP << 'ENDSSH'
cd ~/quantum_trader
docker compose -f docker-compose.vps.yml build api-server
docker compose -f docker-compose.vps.yml build ai-engine
ENDSSH
echo -e "${GREEN}✅ Containers rebuilt${NC}"

# STEG 3: RESTART SERVICES
echo ""
echo -e "${YELLOW}3️⃣ Restarting services...${NC}"
ssh -i $SSH_KEY $VPS_USER@$VPS_IP << 'ENDSSH'
cd ~/quantum_trader
docker compose -f docker-compose.vps.yml restart api-server
docker compose -f docker-compose.vps.yml restart cross-exchange
docker compose -f docker-compose.vps.yml restart eventbus-bridge
ENDSSH
echo -e "${GREEN}✅ Services restarted${NC}"

# STEG 4: WAIT
echo ""
echo -e "${YELLOW}4️⃣ Waiting for startup (15 seconds)...${NC}"
sleep 15

# STEG 5: TEST CIRCUIT BREAKER API
echo ""
echo -e "${YELLOW}5️⃣ Testing Circuit Breaker API...${NC}"
ssh -i $SSH_KEY $VPS_USER@$VPS_IP << 'ENDSSH'
curl -s http://localhost:8000/api/circuit-breaker/status | jq '.'
ENDSSH

# STEG 6: CHECK LOGS
echo ""
echo -e "${YELLOW}6️⃣ Checking logs...${NC}"
ssh -i $SSH_KEY $VPS_USER@$VPS_IP << 'ENDSSH'
echo "=== BACKEND LOGS ==="
docker logs quantum_backend 2>&1 | tail -20 | grep -E "(Circuit Breaker|Redis)" || echo "No matches"

echo ""
echo "=== CROSS EXCHANGE LOGS ==="
docker logs quantum_cross_exchange 2>&1 | tail -20 | grep -E "(Redis|Connected)" || echo "No matches"

echo ""
echo "=== EVENTBUS BRIDGE LOGS ==="
docker logs quantum_eventbus_bridge 2>&1 | tail -20 | grep -E "(Redis|Connected)" || echo "No matches"
ENDSSH

echo ""
echo -e "${GREEN}════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ PHASE 2 DEPLOYMENT COMPLETE!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════${NC}"
echo ""
echo -e "${BLUE}Test Circuit Breaker API:${NC}"
echo "curl http://46.224.116.254:8000/api/circuit-breaker/status"
echo ""
echo -e "${BLUE}Monitor logs:${NC}"
echo "ssh -i $SSH_KEY $VPS_USER@$VPS_IP \"docker logs -f quantum_backend | grep -E '(CIRCUIT|REDIS)'\""
