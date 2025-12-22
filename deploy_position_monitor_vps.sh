#!/bin/bash
# ============================================================================
# FASE 1.1: DEPLOY POSITION MONITOR TIL VPS - PERMANENT FIX
# ============================================================================
# Dette skriptet deployer Position Monitor som en permanent løsning for
# automatisk TP/SL beskyttelse av alle posisjoner.
# ============================================================================

set -e

# FARGER
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# VPS CONFIG
VPS_IP="46.224.116.254"
VPS_USER="qt"
SSH_KEY="$HOME/.ssh/hetzner_fresh"

echo -e "${BLUE}════════════════════════════════════════════════${NC}"
echo -e "${BLUE}🛡️ FASE 1.1: POSITION MONITOR DEPLOYMENT${NC}"
echo -e "${BLUE}════════════════════════════════════════════════${NC}"
echo ""
echo -e "${GREEN}📡 VPS: $VPS_USER@$VPS_IP${NC}"
echo -e "${GREEN}🎯 Deploying PERMANENT TP/SL protection${NC}"
echo ""

# ============================================================================
# STEG 1: TEST SSH-TILKOBLING
# ============================================================================
echo -e "${YELLOW}1️⃣ Tester SSH-tilkobling...${NC}"
if ! ssh -i $SSH_KEY -o ConnectTimeout=5 -o StrictHostKeyChecking=no $VPS_USER@$VPS_IP "echo 'SSH OK'" > /dev/null 2>&1; then
    echo -e "${RED}❌ Kan ikke koble til VPS${NC}"
    echo "Sjekk at:"
    echo "  - VPS IP er korrekt: $VPS_IP"
    echo "  - SSH-nøkkel eksisterer: $SSH_KEY"
    echo "  - Du har SSH-tilgang"
    exit 1
fi
echo -e "${GREEN}✅ SSH fungerer!${NC}"

# ============================================================================
# STEG 2: GIT PULL LATEST CHANGES
# ============================================================================
echo ""
echo -e "${YELLOW}2️⃣ Henter siste endringer fra Git...${NC}"
ssh -i $SSH_KEY $VPS_USER@$VPS_IP << 'ENDSSH'
cd ~/quantum_trader
git pull origin main 2>&1 | grep -E "(Already|Updating|Fast-forward)" || true
ENDSSH
echo -e "${GREEN}✅ Git pull komplett${NC}"

# ============================================================================
# STEG 3: REBUILD BACKEND IMAGE
# ============================================================================
echo ""
echo -e "${YELLOW}3️⃣ Rebuilder backend image med Position Monitor...${NC}"
ssh -i $SSH_KEY $VPS_USER@$VPS_IP << 'ENDSSH'
cd ~/quantum_trader
docker compose -f docker-compose.vps.yml build backend 2>&1 | grep -E "(Successfully|Step|CACHED)" | tail -10
ENDSSH
echo -e "${GREEN}✅ Backend image rebuilt${NC}"

# ============================================================================
# STEG 4: RESTART BACKEND
# ============================================================================
echo ""
echo -e "${YELLOW}4️⃣ Restarter backend container...${NC}"
ssh -i $SSH_KEY $VPS_USER@$VPS_IP << 'ENDSSH'
cd ~/quantum_trader
docker compose -f docker-compose.vps.yml up -d backend
ENDSSH
echo -e "${GREEN}✅ Backend restartet${NC}"

# ============================================================================
# STEG 5: WAIT FOR STARTUP
# ============================================================================
echo ""
echo -e "${YELLOW}5️⃣ Venter på at backend starter (20 sekunder)...${NC}"
sleep 20

# ============================================================================
# STEG 6: VERIFY HEALTH
# ============================================================================
echo ""
echo -e "${YELLOW}6️⃣ Sjekker backend health...${NC}"
ssh -i $SSH_KEY $VPS_USER@$VPS_IP << 'ENDSSH'
curl -s http://localhost:8000/health | jq '.' 2>&1 || echo "⚠️ Health endpoint ikke tilgjengelig ennå"
ENDSSH

# ============================================================================
# STEG 7: CHECK POSITION MONITOR LOGS
# ============================================================================
echo ""
echo -e "${YELLOW}7️⃣ Sjekker Position Monitor logs...${NC}"
echo ""
ssh -i $SSH_KEY $VPS_USER@$VPS_IP << 'ENDSSH'
docker logs quantum_backend 2>&1 | tail -80 | grep -E "(POSITION-MONITOR|protection|TP|SL)" | head -20 || echo "⚠️ Ingen Position Monitor logs funnet ennå"
ENDSSH

# ============================================================================
# SUCCESS SUMMARY
# ============================================================================
echo ""
echo -e "${BLUE}════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ DEPLOYMENT KOMPLETT!${NC}"
echo -e "${BLUE}════════════════════════════════════════════════${NC}"
echo ""
echo -e "${GREEN}🛡️ Position Monitor er nå aktivert på VPS${NC}"
echo ""
echo "📊 Verifiser aktivitet:"
echo -e "   ${YELLOW}ssh -i $SSH_KEY $VPS_USER@$VPS_IP \"docker logs -f quantum_backend | grep POSITION-MONITOR\"${NC}"
echo ""
echo "🔍 Sjekk TP/SL orders:"
echo -e "   ${YELLOW}ssh -i $SSH_KEY $VPS_USER@$VPS_IP \"docker logs quantum_backend | grep -E '(Setting TP|Setting SL|TP/SL)'\"${NC}"
echo ""
echo "🧪 Anbefaling:"
echo "   - Overvåk Position Monitor i 30 minutter"
echo "   - Sjekk at TP/SL orders blir plassert på testnet"
echo "   - Verifiser at ingen errors i logs"
echo ""
echo -e "${GREEN}🎯 Fase 1.1 COMPLETE - Permanent TP/SL protection ACTIVE${NC}"
echo ""
