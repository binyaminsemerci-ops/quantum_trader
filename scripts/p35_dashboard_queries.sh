#!/bin/bash
# P3.5 Decision Intelligence - Actionable Dashboard Queries
# Quick operational visibility into "why not trading"

set -euo pipefail

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

# ============================================================================
# Query A: "Hvorfor handler vi ikke nå?" (5 minutes)
# ============================================================================
query_a() {
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${YELLOW}📊 A) HVORFOR HANDLER VI IKKE NÅ? (5 minutter)${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
    echo ""
    
    echo -e "${CYAN}Decision Distribution:${NC}"
    redis-cli HGETALL quantum:p35:decision:counts:5m | paste -d " " - - | \
        awk '{printf "   %-15s %s\n", $1":", $2}'
    echo ""
    
    echo -e "${CYAN}Top 15 Reasons (with scores):${NC}"
    redis-cli ZREVRANGE quantum:p35:reason:top:5m 0 14 WITHSCORES | paste -d " " - - | \
        awk '{printf "   %-35s %s\n", $1, $2}'
    echo ""
    
    # Calculate skip percentage
    TOTAL=$(redis-cli HGETALL quantum:p35:decision:counts:5m | grep -v ":" | awk '{sum+=$1} END {print sum}')
    SKIP=$(redis-cli HGET quantum:p35:decision:counts:5m SKIP 2>/dev/null || echo "0")
    if [ "$TOTAL" -gt 0 ]; then
        SKIP_PCT=$(echo "scale=1; $SKIP * 100 / $TOTAL" | bc 2>/dev/null || echo "N/A")
        echo -e "${CYAN}Skip Rate:${NC} ${SKIP}/${TOTAL} (${SKIP_PCT}%)"
    fi
    echo ""
}

# ============================================================================
# Query B: "Hva er topp gate per symbol?" (BTC/ETH)
# ============================================================================
query_b() {
    local SYMBOLS="${1:-BTCUSDT ETHUSDT}"
    
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${YELLOW}🎯 B) TOPP GATE PER SYMBOL (5 minutter)${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
    echo ""
    
    for SYMBOL in $SYMBOLS; do
        echo -e "${CYAN}Symbol: ${SYMBOL}${NC}"
        
        # Get all bucket keys from last 5 minutes
        NOW_TS=$(date +%s)
        FIVE_MIN_AGO=$((NOW_TS - 300))
        
        # Extract symbol-specific reasons from buckets
        RESULTS=$(redis-cli --scan --pattern "quantum:p35:bucket:*" | while read bucket; do
            redis-cli HGETALL "$bucket" | grep -A1 "symbol_reason:${SYMBOL}:" 2>/dev/null || true
        done | paste -d " " - - | awk -F: '{print $(NF-1), $NF}' | \
            awk '{reason[$1]+=$2} END {for (r in reason) print reason[r], r}' | \
            sort -rn | head -10)
        
        if [ -z "$RESULTS" ]; then
            echo "   (No data for ${SYMBOL})"
        else
            echo "$RESULTS" | awk '{printf "   %-35s %s\n", $2, $1}'
        fi
        echo ""
    done
}

# ============================================================================
# Query C: "Gate Share" (percentage breakdown)
# ============================================================================
query_c() {
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${YELLOW}📈 C) GATE SHARE (5 minutter) - Top 10 med prosent${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
    echo ""
    
    # Get total decisions
    TOTAL=$(redis-cli HGETALL quantum:p35:decision:counts:5m | grep -v ":" | awk '{sum+=$1} END {print sum}')
    
    if [ "$TOTAL" -eq 0 ]; then
        echo "   (No decisions in last 5 minutes)"
        echo ""
        return
    fi
    
    echo -e "${CYAN}Total Decisions: ${TOTAL}${NC}"
    echo ""
    
    # Get top reasons with share calculation
    redis-cli ZREVRANGE quantum:p35:reason:top:5m 0 9 WITHSCORES | paste -d " " - - | \
        awk -v total="$TOTAL" '{
            share = ($2 / total) * 100
            printf "   %-35s %6s  (%5.1f%%)\n", $1, $2, share
        }'
    echo ""
    
    # Alert if single reason dominates
    TOP_REASON=$(redis-cli ZREVRANGE quantum:p35:reason:top:5m 0 0 WITHSCORES | paste -d " " - -)
    if [ -n "$TOP_REASON" ]; then
        TOP_SCORE=$(echo "$TOP_REASON" | awk '{print $2}')
        TOP_NAME=$(echo "$TOP_REASON" | awk '{print $1}')
        TOP_SHARE=$(echo "scale=1; $TOP_SCORE * 100 / $TOTAL" | bc)
        
        if [ "$(echo "$TOP_SHARE > 40" | bc)" -eq 1 ]; then
            echo -e "${RED}⚠️  ALERT: '${TOP_NAME}' dominates with ${TOP_SHARE}%${NC}"
            echo ""
        fi
    fi
}

# ============================================================================
# Query D: "Drift Detection Light" (gate explosion check)
# ============================================================================
query_d() {
    local THRESHOLD="${1:-40}"  # Default 40% threshold
    
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${YELLOW}🚨 D) DRIFT DETECTION (5 minutter) - Threshold: ${THRESHOLD}%${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
    echo ""
    
    # Get total decisions
    TOTAL=$(redis-cli HGETALL quantum:p35:decision:counts:5m | grep -v ":" | awk '{sum+=$1} END {print sum}')
    
    if [ "$TOTAL" -eq 0 ]; then
        echo "   (No decisions to analyze)"
        echo ""
        return
    fi
    
    # Check top 5 reasons for threshold breach
    ALERTS=0
    redis-cli ZREVRANGE quantum:p35:reason:top:5m 0 4 WITHSCORES | paste -d " " - - | \
        while read REASON SCORE; do
            SHARE=$(echo "scale=1; $SCORE * 100 / $TOTAL" | bc)
            SHARE_INT=$(echo "$SHARE" | cut -d. -f1)
            
            if [ "$SHARE_INT" -ge "$THRESHOLD" ]; then
                echo -e "${RED}🔥 ALERT: ${REASON}${NC}"
                echo "   Share: ${SHARE}% (${SCORE}/${TOTAL})"
                echo "   Action Required: Investigate why this gate is blocking"
                echo ""
                ALERTS=$((ALERTS + 1))
            fi
        done
    
    if [ "$ALERTS" -eq 0 ]; then
        echo -e "${GREEN}✅ No gates exceeding ${THRESHOLD}% threshold${NC}"
        echo ""
    fi
}

# ============================================================================
# Query E: "Service Health" (P3.5 status)
# ============================================================================
query_e() {
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${YELLOW}❤️  E) SERVICE HEALTH${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
    echo ""
    
    echo -e "${CYAN}P3.5 Status:${NC}"
    redis-cli HGETALL quantum:p35:status | paste -d " " - - | \
        awk '{printf "   %-20s %s\n", $1":", $2}'
    echo ""
    
    echo -e "${CYAN}Consumer Group:${NC}"
    PENDING=$(redis-cli XPENDING quantum:stream:apply.result p35_decision_intel 2>/dev/null | head -1)
    echo "   Pending messages: ${PENDING:-0}"
    echo ""
    
    echo -e "${CYAN}Systemd Status:${NC}"
    if systemctl is-active --quiet quantum-p35-decision-intelligence; then
        echo -e "   ${GREEN}✅ Service is ACTIVE${NC}"
    else
        echo -e "   ${RED}❌ Service is NOT ACTIVE${NC}"
    fi
    echo ""
}

# ============================================================================
# Main menu
# ============================================================================
show_menu() {
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}P3.5 DECISION INTELLIGENCE - DASHBOARD QUERIES${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo "Usage: $0 [query]"
    echo ""
    echo "Queries:"
    echo "  a, A     - Hvorfor handler vi ikke nå? (5 min overview)"
    echo "  b, B     - Topp gate per symbol (BTC/ETH)"
    echo "  c, C     - Gate share (percentage breakdown)"
    echo "  d, D     - Drift detection (gate explosion alerts)"
    echo "  e, E     - Service health (P3.5 status)"
    echo "  all      - Run all queries"
    echo ""
    echo "Examples:"
    echo "  $0 a              # Quick overview"
    echo "  $0 b              # Symbol-specific gates (default: BTC/ETH)"
    echo "  $0 b \"BTCUSDT ETHUSDT SOLUSDT\"  # Custom symbols"
    echo "  $0 d              # Check for gate explosions (>40%)"
    echo "  $0 d 50           # Custom threshold (>50%)"
    echo "  $0 all            # Full dashboard"
    echo ""
}

# ============================================================================
# Main execution
# ============================================================================
case "${1:-menu}" in
    a|A)
        query_a
        ;;
    b|B)
        query_b "${2:-}"
        ;;
    c|C)
        query_c
        ;;
    d|D)
        query_d "${2:-}"
        ;;
    e|E)
        query_e
        ;;
    all)
        query_a
        query_b
        query_c
        query_d
        query_e
        ;;
    menu|--help|-h|help)
        show_menu
        ;;
    *)
        echo -e "${RED}Unknown query: $1${NC}"
        echo ""
        show_menu
        exit 1
        ;;
esac
