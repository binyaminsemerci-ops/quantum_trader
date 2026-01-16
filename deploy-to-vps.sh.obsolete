#!/bin/bash
# ============================================================================
# QUANTUM TRADER - KOMPLETT VPS DEPLOYMENT SCRIPT
# ============================================================================
# Dette skriptet gjør ALT automatisk:
# 1. Setup VPS (installer podman, python, etc)
# 2. Clone repository
# 3. Kopier .env og model-filer
# 4. Start services
# 5. Verifiser at alt fungerer
# ============================================================================

set -e  # Exit ved feil

# FARGER
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# ============================================================================
# KONFIGURASJON - ENDRE DISSE!
# ============================================================================
VPS_IP="46.224.116.254"  # Hetzner VPS
VPS_USER="qt"  # VPS brukernavn
SSH_KEY="$HOME/.ssh/hetzner_fresh"  # SSH private key

# ============================================================================
# SJEKK AT VPS_IP ER SATT
# ============================================================================
if [ -z "$VPS_IP" ]; then
    echo -e "${RED}❌ FEIL: VPS_IP er ikke satt!${NC}"
    echo ""
    echo "Åpne dette skriptet og sett VPS_IP på linje 19:"
    echo "VPS_IP=\"din.vps.ip.adresse\""
    echo ""
    exit 1
fi

echo -e "${BLUE}════════════════════════════════════════════════${NC}"
echo -e "${BLUE}🚀 QUANTUM TRADER - VPS DEPLOYMENT${NC}"
echo -e "${BLUE}════════════════════════════════════════════════${NC}"
echo ""
echo -e "${GREEN}📡 VPS: $VPS_USER@$VPS_IP${NC}"
echo ""

# ============================================================================
# STEG 1: TEST SSH-TILKOBLING
# ============================================================================
echo -e "${YELLOW}1️⃣ Tester SSH-tilkobling...${NC}"
if ssh -i $SSH_KEY -o ConnectTimeout=5 -o StrictHostKeyChecking=no $VPS_USER@$VPS_IP "echo 'SSH OK'" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ SSH fungerer!${NC}"
else
    echo -e "${RED}❌ Kan ikke koble til VPS${NC}"
    echo "Sjekk at:"
    echo "  - VPS IP er korrekt"
    echo "  - Du har SSH-tilgang"
    echo "  - SSH-nøkkel er satt opp"
    exit 1
fi

# ============================================================================
# STEG 2: SETUP VPS (INSTALLER SOFTWARE)
# ============================================================================
echo ""
echo -e "${YELLOW}2️⃣ Setter opp VPS (installer podman, python, git)...${NC}"

ssh -i $SSH_KEY $VPS_USER@$VPS_IP << 'ENDSSH'
set -e

echo "📦 Oppdaterer system..."
sudo apt update > /dev/null 2>&1

echo "🐳 Installerer Podman..."
sudo apt install -y podman > /dev/null 2>&1

echo "🐍 Installerer Python..."
sudo apt install -y python3 python3-pip git curl jq > /dev/null 2>&1

echo "📦 Installerer podman-compose..."
pip3 install --user podman-compose > /dev/null 2>&1

echo "🔥 Konfigurerer firewall..."
sudo apt install -y ufw > /dev/null 2>&1
sudo ufw allow ssh > /dev/null 2>&1
sudo ufw allow 8000/tcp > /dev/null 2>&1
sudo ufw allow 8001/tcp > /dev/null 2>&1
sudo ufw --force enable > /dev/null 2>&1

echo "✅ VPS setup komplett!"
ENDSSH

echo -e "${GREEN}✅ VPS setup komplett!${NC}"

# ============================================================================
# STEG 3: CLONE REPOSITORY
# ============================================================================
echo ""
echo -e "${YELLOW}3️⃣ Cloner repository til VPS...${NC}"

ssh -i $SSH_KEY $VPS_USER@$VPS_IP << 'ENDSSH'
set -e

# Fjern gammel kopi hvis den finnes
if [ -d ~/quantum_trader ]; then
    echo "⚠️ Fjerner gammel quantum_trader..."
    rm -rf ~/quantum_trader
fi

# Clone repo
echo "📥 Cloner fra GitHub..."
cd ~
git clone https://github.com/binyaminsemerci-ops/quantum_trader.git > /dev/null 2>&1

echo "✅ Repository clonet!"
ENDSSH

echo -e "${GREEN}✅ Repository clonet!${NC}"

# ============================================================================
# STEG 4: KOPIER .ENV OG MODEL-FILER
# ============================================================================
echo ""
echo -e "${YELLOW}4️⃣ Kopierer .env og model-filer...${NC}"

# Kopier .env
echo "📋 Kopierer .env..."
scp -i $SSH_KEY ~/quantum_trader/.env $VPS_USER@$VPS_IP:~/quantum_trader/ > /dev/null 2>&1
echo -e "${GREEN}✅ .env kopiert${NC}"

# Kopier models (110MB - kan ta litt tid)
echo "🧠 Kopierer AI-modeller (110MB)..."
rsync -az --progress -e "ssh -i $SSH_KEY" ~/quantum_trader/models/ $VPS_USER@$VPS_IP:~/quantum_trader/models/
echo -e "${GREEN}✅ Modeller kopiert${NC}"

# Kopier database (valgfritt)
if [ -d ~/quantum_trader/database ]; then
    echo "💾 Kopierer database..."
    rsync -az --progress -e "ssh -i $SSH_KEY" ~/quantum_trader/database/ $VPS_USER@$VPS_IP:~/quantum_trader/database/
    echo -e "${GREEN}✅ Database kopiert${NC}"
fi

# ============================================================================
# STEG 5: START SERVICES PÅ VPS
# ============================================================================
echo ""
echo -e "${YELLOW}5️⃣ Starter services på VPS...${NC}"

ssh -i $SSH_KEY $VPS_USER@$VPS_IP << 'ENDSSH'
set -e

cd ~/quantum_trader

# Gjør skript kjørbare
chmod +x scripts/*.sh

# Start services
echo "🚀 Starter Redis + AI Engine..."
podman-compose -f docker-compose.wsl.yml up -d redis ai-engine

# Vent litt
sleep 10

echo "✅ Services startet!"
ENDSSH

echo -e "${GREEN}✅ Services startet!${NC}"

# ============================================================================
# STEG 6: VERIFISER DEPLOYMENT
# ============================================================================
echo ""
echo -e "${YELLOW}6️⃣ Verifiserer deployment...${NC}"

ssh -i $SSH_KEY $VPS_USER@$VPS_IP << 'ENDSSH'
set -e

cd ~/quantum_trader

echo "🔍 Sjekker containere..."
podman ps

echo ""
echo "🏥 Tester health endpoints..."

# Test Redis
if podman exec quantum_redis redis-cli ping | grep -q PONG; then
    echo "✅ Redis: HEALTHY"
else
    echo "❌ Redis: UNHEALTHY"
fi

# Test AI Engine
if curl -s http://localhost:8001/health | grep -q "status"; then
    echo "✅ AI Engine: HEALTHY"
else
    echo "❌ AI Engine: UNHEALTHY"
fi

ENDSSH

# ============================================================================
# FERDIG!
# ============================================================================
echo ""
echo -e "${GREEN}════════════════════════════════════════════════${NC}"
echo -e "${GREEN}🎉 DEPLOYMENT KOMPLETT!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════${NC}"
echo ""
echo -e "${BLUE}📊 Neste steg:${NC}"
echo ""
echo "1. SSH til VPS:"
echo -e "   ${YELLOW}ssh $VPS_USER@$VPS_IP${NC}"
echo ""
echo "2. Sjekk logs:"
echo -e "   ${YELLOW}podman logs -f quantum_ai_engine${NC}"
echo ""
echo "3. Test health endpoint:"
echo -e "   ${YELLOW}curl http://localhost:8001/health${NC}"
echo ""
echo "4. Se alle containere:"
echo -e "   ${YELLOW}podman ps${NC}"
echo ""
echo -e "${GREEN}🚀 Quantum Trader kjører nå på VPS!${NC}"
echo ""
