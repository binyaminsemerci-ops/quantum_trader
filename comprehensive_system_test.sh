#!/bin/bash

###############################################################################
# AI HEDGE FUND OS - HELHETLIG TESTPLAN
# Comprehensive System Validation Script
# Versjon: 1.0
# Dato: 2025-12-20
###############################################################################

set -e

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
        echo -e "${GREEN}Systemet er verifisert funksjonelt og stabilt i dry-run mode.${NC}"
        return 0
    else
        echo -e "\n${RED}❌ NOEN TESTER FEILET${NC}"
        echo -e "${RED}Vennligst gå gjennom feilene ovenfor før du fortsetter.${NC}"
        return 1
    fi
}

###############################################################################
# TRINN 1 – KONTROLLER SERVICE-HELSE
###############################################################################
test_container_health() {
    log_section "TRINN 1 – SERVICE HELSE"
    
    log_info "Sjekker systemd services..."
    
    # Required services
    REQUIRED_CONTAINERS=(
        "quantum-backend"
        "quantum-trading-bot"
        "quantum-redis"
        "quantum-rl-optimizer"
        "quantum-strategy-evaluator"
        "quantum-strategy-evolution"
        "quantum-policy-memory"
        "quantum-auto-executor"
        "quantum-federation-stub"
    )
    
    # Display service status
    echo ""
    systemctl list-units 'quantum-*.service' --no-pager --no-legend | head -20 || {
        log_error "systemd er ikke tilgjengelig"
        return 1
    }
    echo ""
    
    # Check each required service
    for container in "${REQUIRED_CONTAINERS[@]}"; do
        if systemctl is-active "${container}.service" >/dev/null 2>&1; then
            STATUS=$(systemctl is-active "${container}.service")
            if [ "$STATUS" = "active" ]; then
                log_success "Service '${container}' er oppe og kjører"
            else
                log_error "Service '${container}' kjører ikke (Status: $STATUS)"
            fi
        else
            log_error "Service '${container}' finnes ikke"
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
        echo "$BACKEND_RESPONSE" | python3 -m json.tool
        
        if echo "$BACKEND_RESPONSE" | grep -q '"status".*"ok"'; then
            log_success "Backend API returnerer status 'ok'"
        else
            log_error "Backend API returnerer ikke 'ok' status"
        fi
    else
        log_error "Kunne ikke nå backend API på http://localhost:8000/health"
    fi
    
    echo ""
    
    # Test AI Engine Health
    log_info "Testing AI Engine health endpoint..."
    if AI_RESPONSE=$(curl -s -f http://localhost:8001/health 2>/dev/null); then
        echo "$AI_RESPONSE" | python3 -m json.tool
        
        if echo "$AI_RESPONSE" | grep -q '"status".*"ok"'; then
            log_success "AI Engine API returnerer status 'ok'"
        else
            log_error "AI Engine API returnerer ikke 'ok' status"
        fi
        
        # Check for loaded models
        if echo "$AI_RESPONSE" | grep -q "models"; then
            log_success "AI Engine rapporterer lastede modeller"
        else
            log_warning "AI Engine rapporterer ingen modeller"
        fi
    else
        log_error "Kunne ikke nå AI Engine API på http://localhost:8001/health"
    fi
}

###############################################################################
# TRINN 3 – REDIS DATAINTEGRITET
###############################################################################
test_redis_integrity() {
    log_section "TRINN 3 – REDIS DATAINTEGRITET"
    
    # Check Redis memory usage
    log_info "Sjekker Redis minnebruk..."
    MEMORY=$(redis-cli info memory 2>/dev/null | grep used_memory_human | cut -d: -f2 | tr -d '\r')
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
        "current_policy"
        "meta_best_strategy"
        "quantum_regime_forecast"
        "system_ssi"
    )
    
    ALL_KEYS=$(redis-cli keys "*" 2>/dev/null)
    
    for key in "${REQUIRED_KEYS[@]}"; do
        if echo "$ALL_KEYS" | grep -q "$key"; then
            log_success "Nøkkel '$key' finnes i Redis"
        else
            log_warning "Nøkkel '$key' mangler i Redis (kan være normalt ved oppstart)"
        fi
    done
    
    echo ""
    log_info "Alle Redis nøkler:"
    echo "$ALL_KEYS"
}

###############################################################################
# TRINN 4 – AI-MODELL SANITY-CHECK
###############################################################################
test_ai_models() {
    log_section "TRINN 4 – AI-MODELL SANITY-CHECK"
    
    log_info "Tester AI-modeller (xgb, lgbm, nhits, patchtst)..."
    
    # Test model initialization
    MODEL_TEST=$(docker exec quantum_ai_engine python3 -c "
try:
    from ai_engine.ensemble_manager import EnsembleManager
    e = EnsembleManager(enabled_models=['xgb','lgbm','nhits','patchtst'])
    result = {m: getattr(e, f'{m}_agent', None) is not None for m in e.enabled_models}
    print(result)
    exit(0)
except Exception as ex:
    print(f'ERROR: {ex}')
    exit(1)
" 2>&1)
    
    if echo "$MODEL_TEST" | grep -q "ERROR"; then
        log_error "AI-modell test feilet: $MODEL_TEST"
    else
        echo "$MODEL_TEST"
        
        # Check each model
        for model in xgb lgbm nhits patchtst; do
            if echo "$MODEL_TEST" | grep -q "'$model': True"; then
                log_success "Modell '$model' er lastet og tilgjengelig"
            else
                log_error "Modell '$model' er ikke lastet korrekt"
            fi
        done
    fi
}

###############################################################################
# TRINN 5 – REGIME-FORECAST VALIDERING
###############################################################################
test_regime_forecast() {
    log_section "TRINN 5 – REGIME-FORECAST VALIDERING"
    
    log_info "Sjekker quantum_regime_forecast..."
    
    FORECAST=$(redis-cli hgetall quantum_regime_forecast 2>/dev/null)
    
    if [ -n "$FORECAST" ]; then
        echo "$FORECAST"
        
        # Check for timestamp
        if echo "$FORECAST" | grep -q "timestamp"; then
            TIMESTAMP=$(echo "$FORECAST" | grep -A1 "timestamp" | tail -1)
            log_success "Regime forecast har tidsstempel: $TIMESTAMP"
            
            # Check if timestamp is recent (within 6 hours = 21600 seconds)
            CURRENT_TIME=$(date +%s)
            FORECAST_TIME=$(echo "$TIMESTAMP" | xargs)
            
            if [ -n "$FORECAST_TIME" ] && [ "$FORECAST_TIME" != "timestamp" ]; then
                TIME_DIFF=$((CURRENT_TIME - FORECAST_TIME))
                if [ $TIME_DIFF -lt 21600 ]; then
                    log_success "Forecast er nylig oppdatert (for $((TIME_DIFF / 60)) minutter siden)"
                else
                    log_warning "Forecast er gammel (for $((TIME_DIFF / 3600)) timer siden)"
                fi
            fi
        else
            log_warning "Ingen tidsstempel funnet i regime forecast"
        fi
        
        # Check for regime probabilities
        if echo "$FORECAST" | grep -qE "(bull|bear|neutral|volatile)"; then
            log_success "Regime forecast inneholder regime-sannsynligheter"
        else
            log_warning "Ingen regime-sannsynligheter funnet"
        fi
    else
        log_warning "quantum_regime_forecast finnes ikke eller er tom"
    fi
}

###############################################################################
# TRINN 6 – GOVERNANCE OG SSI
###############################################################################
test_governance_ssi() {
    log_section "TRINN 6 – GOVERNANCE OG SSI"
    
    # Check System Stress Index
    log_info "Sjekker System Stress Index (SSI)..."
    SSI=$(redis-cli get system_ssi 2>/dev/null)
    
    if [ -n "$SSI" ]; then
        log_success "SSI verdi: $SSI"
        
        # Validate SSI range (-2 to 2)
        if command -v python3 &> /dev/null; then
            IN_RANGE=$(python3 -c "print(-2 <= float('$SSI') <= 2)" 2>/dev/null || echo "False")
            if [ "$IN_RANGE" = "True" ]; then
                log_success "SSI er innenfor gyldig område (-2 til 2)"
            else
                log_warning "SSI er utenfor normalt område (-2 til 2)"
            fi
        fi
    else
        log_warning "system_ssi finnes ikke i Redis"
    fi
    
    echo ""
    
    # Check Governance Weights
    log_info "Sjekker governance_weights..."
    WEIGHTS=$(redis-cli hgetall governance_weights 2>/dev/null)
    
    if [ -n "$WEIGHTS" ]; then
        echo "$WEIGHTS"
        log_success "Governance weights finnes"
        
        # Optionally validate that weights sum to ~1
        if command -v python3 &> /dev/null; then
            WEIGHT_SUM=$(echo "$WEIGHTS" | python3 -c "
import sys
lines = sys.stdin.readlines()
values = [float(lines[i].strip()) for i in range(1, len(lines), 2) if lines[i].strip().replace('.','').replace('-','').isdigit()]
print(sum(values))
" 2>/dev/null || echo "0")
            
            if [ -n "$WEIGHT_SUM" ]; then
                log_info "Sum av vekter: $WEIGHT_SUM"
                # Check if sum is approximately 1 (within 0.1)
                VALID_SUM=$(python3 -c "print(0.9 <= float('$WEIGHT_SUM') <= 1.1)" 2>/dev/null || echo "False")
                if [ "$VALID_SUM" = "True" ]; then
                    log_success "Vekter summerer til ~1.0"
                else
                    log_warning "Vekter summerer ikke til ~1.0 (sum: $WEIGHT_SUM)"
                fi
            fi
        fi
    else
        log_warning "governance_weights finnes ikke i Redis"
    fi
}

###############################################################################
# TRINN 7 – FULL END-TO-END SIMULERING
###############################################################################
test_end_to_end() {
    log_section "TRINN 7 – END-TO-END SIMULERING"
    
    log_info "Sender syntetiske signal-kall for BTC, ETH, SOL..."
    
    SYMBOLS=("BTCUSDT" "ETHUSDT" "SOLUSDT")
    
    for symbol in "${SYMBOLS[@]}"; do
        echo ""
        log_info "Testing signal for $symbol..."
        
        RESPONSE=$(curl -s -X POST http://localhost:8001/api/ai/signal \
            -H "Content-Type: application/json" \
            -d "{\"symbol\":\"$symbol\"}" 2>/dev/null)
        
        if [ -n "$RESPONSE" ]; then
            echo "$RESPONSE" | jq '.' 2>/dev/null || echo "$RESPONSE"
            
            # Extract action and confidence
            ACTION=$(echo "$RESPONSE" | jq -r '.action' 2>/dev/null)
            CONFIDENCE=$(echo "$RESPONSE" | jq -r '.confidence' 2>/dev/null)
            
            if [ -n "$ACTION" ] && [ "$ACTION" != "null" ]; then
                if [[ "$ACTION" =~ ^(BUY|SELL|HOLD)$ ]]; then
                    log_success "$symbol: Action=$ACTION, Confidence=$CONFIDENCE"
                    
                    # Check confidence level
                    if [ -n "$CONFIDENCE" ] && [ "$CONFIDENCE" != "null" ]; then
                        CONF_OK=$(python3 -c "print(float('$CONFIDENCE') > 0.4)" 2>/dev/null || echo "False")
                        if [ "$CONF_OK" = "True" ]; then
                            log_success "$symbol: Confidence > 0.4 ✓"
                        else
                            log_warning "$symbol: Confidence < 0.4"
                        fi
                    fi
                else
                    log_warning "$symbol: Uventet action: $ACTION"
                fi
            else
                log_error "$symbol: Ingen gyldig action i response"
            fi
        else
            log_error "$symbol: Ingen response fra API"
        fi
    done
}

###############################################################################
# TRINN 8 – EVALUÉR LOGGENE
###############################################################################
test_logs() {
    log_section "TRINN 8 – LOGG-EVALUERING"
    
    log_info "Søker etter kritiske feil i logger (siste 1000 linjer)..."
    
    ERROR_COUNT=$(journalctl --since '1 hour ago' --no-pager | grep -c -E "ERROR|CRITICAL|Exception" || echo "0")
    
    if [ "$ERROR_COUNT" -eq 0 ]; then
        log_success "Ingen kritiske feil funnet i logger"
    else
        log_warning "Fant $ERROR_COUNT linjer med ERROR/CRITICAL/Exception"
        echo ""
        log_info "Viser de siste 10 feilene:"
        journalctl --since '1 hour ago' --no-pager | grep -E "ERROR|CRITICAL|Exception" | tail -10
    fi
}

###############################################################################
# MAIN EXECUTION
###############################################################################
main() {
    log_section "AI HEDGE FUND OS - HELHETLIG TESTPLAN"
    log_info "Starter omfattende systemvalidering..."
    log_info "Dato: $(date)"
    
    # Run all tests
    test_container_health
    test_internal_apis
    test_redis_integrity
    test_ai_models
    test_regime_forecast
    test_governance_ssi
    test_end_to_end
    test_logs
    
    # Print summary
    print_summary
    RESULT=$?
    
    if [ $RESULT -eq 0 ]; then
        log_section "🏁 NESTE STEG: MANUELL TILKOBLING TIL BINANCE TESTNET"
        echo ""
        echo "For å koble til Binance Testnet:"
        echo ""
        echo "1. Gå til https://testnet.binance.vision og opprett testnet API-nøkler"
        echo ""
        echo "2. Oppdater .env-filen med dine testnet-credentials:"
        echo "   BINANCE_API_KEY=din_testnet_key"
        echo "   BINANCE_API_SECRET=din_testnet_secret"
        echo "   BINANCE_BASE_URL=https://testnet.binance.vision/api"
        echo "   MODE=testnet"
        echo ""
        echo "3. Start systemet på nytt:"
        echo "   sudo systemctl restart quantum-*.service"
        echo ""
        echo -e "${YELLOW}⚠️  VIKTIG: Bruk kun små posisjoner på testnet!${NC}"
        echo -e "${YELLOW}⚠️  Aldri bruk live-nøkler før full bekreftet oppførsel!${NC}"
        echo ""
    fi
    
    return $RESULT
}

# Run main function
main
exit $?
