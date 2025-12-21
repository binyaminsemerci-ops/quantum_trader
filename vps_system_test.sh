#!/bin/bash

###############################################################################
# AI HEDGE FUND OS - VPS HELHETLIG TESTPLAN
# Tilpasset for faktisk VPS deployment
# Versjon: 1.0 VPS
# Dato: 2025-12-20
###############################################################################

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test results tracking
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Log function
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
    ((PASSED_TESTS++))
    ((TOTAL_TESTS++))
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
    ((FAILED_TESTS++))
    ((TOTAL_TESTS++))
}

log_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

log_section() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
}

# Test result summary
print_summary() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  TEST SAMMENDRAG${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "Total tester: ${TOTAL_TESTS}"
    echo -e "${GREEN}Bestått: ${PASSED_TESTS}${NC}"
    echo -e "${RED}Feilet: ${FAILED_TESTS}${NC}"
    
    if [ $FAILED_TESTS -eq 0 ]; then
        echo -e "\n${GREEN}✅ ALLE TESTER BESTÅTT!${NC}"
        echo -e "${GREEN}Systemet er verifisert funksjonelt og stabilt på VPS.${NC}"
        return 0
    else
        echo -e "\n${RED}❌ NOEN TESTER FEILET${NC}"
        echo -e "${RED}Vennligst gå gjennom feilene ovenfor før du fortsetter.${NC}"
        return 1
    fi
}

###############################################################################
# TRINN 1 – KONTROLLER CONTAINER-HELSE
###############################################################################
test_container_health() {
    log_section "TRINN 1 – CONTAINER HELSE"
    
    log_info "Sjekker docker containers..."
    
    # Core containers (kritiske)
    CORE_CONTAINERS=(
        "quantum_backend"
        "quantum_redis"
        "quantum_postgres"
    )
    
    # AI/Trading containers
    AI_CONTAINERS=(
        "quantum_trading_bot"
        "quantum_rl_optimizer"
        "quantum_strategy_evaluator"
        "quantum_strategy_evolution"
        "quantum_auto_executor"
    )
    
    # Orchestration containers
    ORCHESTRATION_CONTAINERS=(
        "quantum_policy_memory"
        "quantum_federation_stub"
    )
    
    # Display container status
    echo ""
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || {
        log_error "Docker er ikke tilgjengelig eller kjører ikke"
        return 1
    }
    echo ""
    
    # Check core containers
    log_info "Sjekker kjerne-containere..."
    for container in "${CORE_CONTAINERS[@]}"; do
        if docker ps --format '{{.Names}}' | grep -q "^${container}$"; then
            STATUS=$(docker ps --filter "name=^${container}$" --format '{{.Status}}')
            if echo "$STATUS" | grep -q "Up"; then
                log_success "✓ Core: '${container}' er oppe"
            else
                log_error "✗ Core: '${container}' kjører ikke (Status: $STATUS)"
            fi
        else
            log_error "✗ Core: '${container}' finnes ikke"
        fi
    done
    
    echo ""
    log_info "Sjekker AI/Trading containere..."
    for container in "${AI_CONTAINERS[@]}"; do
        if docker ps --format '{{.Names}}' | grep -q "^${container}$"; then
            STATUS=$(docker ps --filter "name=^${container}$" --format '{{.Status}}')
            if echo "$STATUS" | grep -q "Up"; then
                log_success "✓ AI: '${container}' er oppe"
            else
                log_error "✗ AI: '${container}' kjører ikke"
            fi
        else
            log_warning "⚠ AI: '${container}' finnes ikke (kan være optional)"
        fi
    done
    
    echo ""
    log_info "Sjekker Orchestration containere..."
    for container in "${ORCHESTRATION_CONTAINERS[@]}"; do
        if docker ps --format '{{.Names}}' | grep -q "^${container}$"; then
            log_success "✓ Orchestration: '${container}' er oppe"
        else
            log_warning "⚠ Orchestration: '${container}' finnes ikke"
        fi
    done
}

###############################################################################
# TRINN 2 – VALIDER INTERNE API-ER
###############################################################################
test_internal_apis() {
    log_section "TRINN 2 – INTERNE API-ER"
    
    # Test Backend Health
    log_info "Testing backend health endpoint..."
    if BACKEND_RESPONSE=$(curl -s -f http://localhost:8000/health 2>/dev/null); then
        echo "$BACKEND_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$BACKEND_RESPONSE"
        
        if echo "$BACKEND_RESPONSE" | grep -q '"status".*"ok"'; then
            log_success "Backend API returnerer status 'ok'"
        else
            log_error "Backend API returnerer ikke 'ok' status"
        fi
    else
        log_error "Kunne ikke nå backend API på http://localhost:8000/health"
    fi
    
    echo ""
    
    # Test Trading Bot Health (replacing AI Engine)
    log_info "Testing Trading Bot health endpoint..."
    if TRADING_RESPONSE=$(curl -s -f http://localhost:8003/health 2>/dev/null); then
        echo "$TRADING_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$TRADING_RESPONSE"
        
        if echo "$TRADING_RESPONSE" | grep -q '"status".*"ok"'; then
            log_success "Trading Bot API returnerer status 'ok'"
        else
            log_warning "Trading Bot API returnerer ikke 'ok' status"
        fi
    else
        log_warning "Kunne ikke nå Trading Bot API på http://localhost:8003/health"
    fi
}

###############################################################################
# TRINN 3 – REDIS DATAINTEGRITET
###############################################################################
test_redis_integrity() {
    log_section "TRINN 3 – REDIS DATAINTEGRITET"
    
    # Check Redis memory usage
    log_info "Sjekker Redis minnebruk..."
    MEMORY=$(docker exec quantum_redis redis-cli info memory 2>/dev/null | grep used_memory_human | cut -d: -f2 | tr -d '\r')
    if [ -n "$MEMORY" ]; then
        log_success "Redis minnebruk: $MEMORY"
    else
        log_error "Kunne ikke hente Redis minnebruk"
    fi
    
    echo ""
    
    # Check for required keys
    log_info "Sjekker tilstedeværelse av kritiske nøkler..."
    
    REQUIRED_KEYS=(
        "governance_weights"
        "meta_best_strategy"
        "governance_active"
    )
    
    ALL_KEYS=$(docker exec quantum_redis redis-cli keys "*" 2>/dev/null)
    
    echo ""
    log_info "Totalt antall nøkler: $(echo "$ALL_KEYS" | wc -l)"
    
    for key in "${REQUIRED_KEYS[@]}"; do
        if echo "$ALL_KEYS" | grep -q "$key"; then
            log_success "Nøkkel '$key' finnes i Redis"
        else
            log_warning "Nøkkel '$key' mangler i Redis (kan være normalt)"
        fi
    done
}

###############################################################################
# TRINN 4 – DATABASE FORBINDELSE
###############################################################################
test_database() {
    log_section "TRINN 4 – DATABASE FORBINDELSE"
    
    log_info "Sjekker PostgreSQL forbindelse..."
    
    if docker exec quantum_postgres pg_isready -U trading_bot &>/dev/null; then
        log_success "PostgreSQL er tilgjengelig"
        
        # Count tables
        TABLE_COUNT=$(docker exec quantum_postgres psql -U trading_bot -d trading_bot -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public';" 2>/dev/null | tr -d ' ')
        if [ -n "$TABLE_COUNT" ]; then
            log_success "Database har $TABLE_COUNT tabeller"
        fi
    else
        log_error "PostgreSQL er ikke tilgjengelig"
    fi
}

###############################################################################
# TRINN 5 – GOVERNANCE OG METRICS
###############################################################################
test_governance() {
    log_section "TRINN 5 – GOVERNANCE OG METRICS"
    
    # Check Governance Weights
    log_info "Sjekker governance_weights..."
    WEIGHTS=$(docker exec quantum_redis redis-cli hgetall governance_weights 2>/dev/null)
    
    if [ -n "$WEIGHTS" ]; then
        echo "$WEIGHTS" | head -10
        log_success "Governance weights finnes"
    else
        log_warning "governance_weights finnes ikke i Redis"
    fi
    
    echo ""
    
    # Check active governance
    log_info "Sjekker governance_active status..."
    ACTIVE=$(docker exec quantum_redis redis-cli get governance_active 2>/dev/null)
    if [ "$ACTIVE" = "true" ]; then
        log_success "Governance er AKTIV"
    else
        log_warning "Governance er ikke aktiv (governance_active=$ACTIVE)"
    fi
}

###############################################################################
# TRINN 6 – SYSTEM METRICS
###############################################################################
test_system_metrics() {
    log_section "TRINN 6 – SYSTEM METRICS"
    
    log_info "Sjekker system metrics..."
    
    # Check latest metrics
    METRICS=$(docker exec quantum_redis redis-cli get latest_metrics 2>/dev/null)
    if [ -n "$METRICS" ]; then
        log_success "Latest metrics finnes"
        echo "$METRICS" | python3 -m json.tool 2>/dev/null | head -15 || echo "$METRICS"
    else
        log_warning "Ingen latest_metrics funnet"
    fi
}

###############################################################################
# TRINN 7 – EVALUÉR LOGGENE
###############################################################################
test_logs() {
    log_section "TRINN 7 – LOGG-EVALUERING"
    
    log_info "Søker etter kritiske feil i backend logger..."
    
    ERROR_COUNT=$(docker logs quantum_backend --tail=500 2>&1 | grep -c -E "ERROR|CRITICAL|Exception" || echo "0")
    
    if [ "$ERROR_COUNT" -eq 0 ]; then
        log_success "Ingen kritiske feil funnet i backend logger"
    else
        log_warning "Fant $ERROR_COUNT linjer med ERROR/CRITICAL/Exception i backend"
        echo ""
        log_info "Viser de siste 5 feilene:"
        docker logs quantum_backend --tail=500 2>&1 | grep -E "ERROR|CRITICAL|Exception" | tail -5
    fi
}

###############################################################################
# TRINN 8 – DISK OG RESSURSER
###############################################################################
test_resources() {
    log_section "TRINN 8 – DISK OG RESSURSER"
    
    log_info "Sjekker disk space..."
    df -h / | tail -1
    
    DISK_USAGE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
    if [ "$DISK_USAGE" -lt 80 ]; then
        log_success "Disk usage er OK ($DISK_USAGE%)"
    elif [ "$DISK_USAGE" -lt 90 ]; then
        log_warning "Disk usage er høy ($DISK_USAGE%)"
    else
        log_error "Disk usage er kritisk ($DISK_USAGE%)"
    fi
    
    echo ""
    log_info "Sjekker minne..."
    free -h | grep Mem:
}

###############################################################################
# MAIN EXECUTION
###############################################################################
main() {
    log_section "AI HEDGE FUND OS - VPS HELHETLIG TESTPLAN"
    log_info "Starter omfattende systemvalidering på VPS..."
    log_info "Dato: $(date)"
    
    # Run all tests
    test_container_health
    test_internal_apis
    test_redis_integrity
    test_database
    test_governance
    test_system_metrics
    test_logs
    test_resources
    
    # Print summary
    print_summary
    RESULT=$?
    
    if [ $RESULT -eq 0 ]; then
        log_section "✅ VPS SYSTEM STATUS: OPERASJONELL"
        echo ""
        echo -e "${GREEN}Systemet kjører stabilt på VPS!${NC}"
        echo ""
    fi
    
    return $RESULT
}

# Run main function
main
exit $?
