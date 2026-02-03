#!/bin/bash
# Quantum Trader - WSL Podman Verification Script
# Purpose: Verify all services are running correctly
# Created: 2025-12-16

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}🔍 Verifying Quantum Trader Services${NC}"
echo "========================================"

# 1️⃣ Check containers are running
echo ""
echo "1️⃣ Container Status:"
podman ps --filter "name=quantum" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" || {
    echo -e "${RED}❌ No containers running${NC}"
    exit 1
}

# 2️⃣ Check Redis
echo ""
echo "2️⃣ Redis Health:"
if podman exec quantum_redis redis-cli ping | grep -q PONG; then
    echo -e "${GREEN}✅ Redis: HEALTHY${NC}"
else
    echo -e "${RED}❌ Redis: UNHEALTHY${NC}"
fi

# 3️⃣ Check AI Engine health endpoint
echo ""
echo "3️⃣ AI Engine Health:"
if curl -s http://localhost:8001/health | grep -q "status"; then
    echo -e "${GREEN}✅ AI Engine: HEALTHY${NC}"
    echo "Response:"
    curl -s http://localhost:8001/health | jq '.' 2>/dev/null || curl -s http://localhost:8001/health
else
    echo -e "${RED}❌ AI Engine: UNHEALTHY${NC}"
    echo "Logs:"
    podman logs --tail 20 quantum_ai_engine
fi

# 4️⃣ Check for import errors in logs
echo ""
echo "4️⃣ Checking for Import Errors:"
if podman logs quantum_ai_engine 2>&1 | grep -i "importerror\|modulenotfounderror\|/mnt/c"; then
    echo -e "${RED}⚠️ Found import issues in logs${NC}"
else
    echo -e "${GREEN}✅ No import errors detected${NC}"
fi

# 5️⃣ Check PYTHONPATH
echo ""
echo "5️⃣ Verifying PYTHONPATH:"
PYTHONPATH_CHECK=$(podman exec quantum_ai_engine env | grep PYTHONPATH || echo "PYTHONPATH not set")
if echo "$PYTHONPATH_CHECK" | grep -q "/app"; then
    echo -e "${GREEN}✅ PYTHONPATH: $PYTHONPATH_CHECK${NC}"
else
    echo -e "${YELLOW}⚠️ PYTHONPATH: $PYTHONPATH_CHECK${NC}"
fi

# 6️⃣ Check for /mnt/c in container
echo ""
echo "6️⃣ Checking for /mnt/c paths:"
if podman exec quantum_ai_engine python3 -c "import sys; print('\n'.join(sys.path))" | grep -q "/mnt/c"; then
    echo -e "${RED}❌ WARNING: /mnt/c found in Python path!${NC}"
    podman exec quantum_ai_engine python3 -c "import sys; print('\n'.join(sys.path))"
else
    echo -e "${GREEN}✅ No /mnt/c paths in Python${NC}"
fi

# 7️⃣ Test ServiceHealth import
echo ""
echo "7️⃣ Testing ServiceHealth import:"
if podman exec quantum_ai_engine python3 -c "from microservices.ai_engine.service_health import ServiceHealth; print('Import successful')" 2>/dev/null; then
    echo -e "${GREEN}✅ ServiceHealth import: SUCCESS${NC}"
else
    echo -e "${RED}❌ ServiceHealth import: FAILED${NC}"
    echo "Attempting import test:"
    podman exec quantum_ai_engine python3 -c "from microservices.ai_engine.service_health import ServiceHealth; print('Import successful')" 2>&1 || true
fi

echo ""
echo -e "${GREEN}🎯 Verification Complete!${NC}"
echo ""
echo "Commands for troubleshooting:"
echo "  - View logs: podman logs -f quantum_ai_engine"
echo "  - Enter container: podman exec -it quantum_ai_engine bash"
echo "  - Restart: podman-compose -f docker-compose.wsl.yml restart ai-engine"
echo "  - Stop all: podman-compose -f docker-compose.wsl.yml down"
