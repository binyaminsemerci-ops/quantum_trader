#!/bin/bash
# Monitor all AI modules health

echo "🔍 AI MODULES HEALTH CHECK"
echo "=========================="
echo ""

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to check service health
check_service() {
    local service=$1
    local port=$2
    
    if [ -z "$port" ]; then
        # Check if container is running
        if docker ps --filter "name=$service" --format "{{.Status}}" | grep -q "Up"; then
            echo -e "${GREEN}✅ $service: Running${NC}"
        else
            echo -e "${RED}❌ $service: Not running${NC}"
        fi
    else
        # Check HTTP health endpoint
        response=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$port/health" 2>/dev/null)
        if [ "$response" = "200" ]; then
            echo -e "${GREEN}✅ $service: Healthy (port $port)${NC}"
        elif [ "$response" = "503" ]; then
            echo -e "${YELLOW}⚠️  $service: Degraded (port $port)${NC}"
        else
            echo -e "${RED}❌ $service: Unhealthy (port $port)${NC}"
        fi
    fi
}

echo "Core Services:"
check_service "quantum_ai_engine" 8001
check_service "quantum_execution" 8002
check_service "quantum_trading_bot" 8003
check_service "quantum_clm"
check_service "quantum_risk_safety" 8005

echo ""
echo "Phase 1 - Observation:"
check_service "quantum_universe_os" 8006
check_service "quantum_model_supervisor" 8007
check_service "quantum_pil"

echo ""
echo "Phase 2 - Portfolio & Risk:"
check_service "quantum_pba" 8008
check_service "quantum_self_healing" 8009
check_service "quantum_orchestrator_policy"

echo ""
echo "Phase 3 - Amplification:"
check_service "quantum_pal" 8010
check_service "quantum_trading_mathematician"

echo ""
echo "Phase 4 - Coordination:"
check_service "quantum_ai_hfos" 8011

echo ""
echo "Phase 5 - Advanced:"
check_service "quantum_ess" 8012
check_service "quantum_aelm"
check_service "quantum_msc_ai"
check_service "quantum_opportunity_ranker"

echo ""
echo "Infrastructure:"
check_service "quantum_redis" 6379
check_service "quantum_postgres" 5432
check_service "quantum_grafana" 3001
check_service "quantum_prometheus" 9090

echo ""
echo "=========================="
echo "Monitor complete!"
