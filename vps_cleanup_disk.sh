#!/bin/bash
###############################################################################
# VPS Disk Cleanup Script
# Frigjør disk space på VPS
###############################################################################

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  VPS DISK CLEANUP${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
echo ""

# Vis current disk usage
echo -e "${YELLOW}BEFORE CLEANUP:${NC}"
df -h / | tail -1
echo ""
docker system df
echo ""

# Ask for confirmation
read -p "$(echo -e ${YELLOW}Vil du rydde opp ubrukte Docker ressurser? [y/N]: ${NC})" -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}Avbrutt.${NC}"
    exit 0
fi

echo ""
echo -e "${BLUE}Starter opprydding...${NC}"
echo ""

# 1. Remove unused images
echo -e "${YELLOW}[1/4] Fjerner ubrukte Docker images...${NC}"
docker image prune -a -f
echo ""

# 2. Remove build cache
echo -e "${YELLOW}[2/4] Fjerner build cache...${NC}"
docker builder prune -a -f
echo ""

# 3. Remove unused volumes
echo -e "${YELLOW}[3/4] Fjerner ubrukte volumes...${NC}"
docker volume prune -f
echo ""

# 4. Remove unused networks
echo -e "${YELLOW}[4/4] Fjerner ubrukte networks...${NC}"
docker network prune -f
echo ""

# Show results
echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  CLEANUP COMPLETE!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
echo ""

echo -e "${YELLOW}AFTER CLEANUP:${NC}"
df -h / | tail -1
echo ""
docker system df
echo ""

# Calculate savings
BEFORE=$(df / | tail -1 | awk '{print $3}')
AFTER=$(df / | tail -1 | awk '{print $3}')
echo -e "${GREEN}✓ Cleanup fullført!${NC}"
