#!/bin/bash
#
# monitor_harvest_brain.sh - Real-time monitoring dashboard for HarvestBrain
# Usage: bash monitor_harvest_brain.sh
#

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

while true; do
    clear
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║         HarvestBrain Monitoring Dashboard                         ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    # Service Status
    echo -e "${YELLOW}=== Service Status ===${NC}"
    if systemctl is-active --quiet quantum-harvest-brain; then
        echo -e "  Status: ${GREEN}✅ ACTIVE${NC}"
        UPTIME=$(systemctl show -p ActiveEnterTimestamp quantum-harvest-brain | cut -d= -f2)
        echo "  Uptime: $UPTIME"
    else
        echo -e "  Status: ${RED}❌ INACTIVE${NC}"
    fi
    
    # Memory & CPU
    MEM=$(ps aux | grep harvest_brain.py | grep -v grep | awk '{print $6/1024 " MB"}')
    CPU=$(ps aux | grep harvest_brain.py | grep -v grep | awk '{print $3 "%"}')
    echo "  Memory: $MEM"
    echo "  CPU: $CPU"
    echo ""
    
    # Configuration
    echo -e "${YELLOW}=== Configuration ===${NC}"
    MODE=$(grep "^HARVEST_MODE" /etc/quantum/harvest-brain.env 2>/dev/null | cut -d= -f2)
    MIN_R=$(grep "^HARVEST_MIN_R" /etc/quantum/harvest-brain.env 2>/dev/null | cut -d= -f2)
    LADDER=$(grep "^HARVEST_LADDER" /etc/quantum/harvest-brain.env 2>/dev/null | cut -d= -f2)
    KILL_SWITCH=$(redis-cli GET quantum:kill 2>/dev/null)
    
    if [[ "$MODE" == "shadow" ]]; then
        echo -e "  Mode: ${YELLOW}SHADOW${NC} (safe, no live orders)"
    else
        echo -e "  Mode: ${GREEN}LIVE${NC} (publishing to trade.intent)"
    fi
    echo "  Min R: $MIN_R"
    echo "  Ladder: $LADDER"
    
    if [[ "$KILL_SWITCH" == "1" ]]; then
        echo -e "  Kill-Switch: ${RED}ACTIVE${NC} (publishing disabled)"
    else
        echo -e "  Kill-Switch: ${GREEN}OFF${NC} (publishing enabled)"
    fi
    echo ""
    
    # Stream Metrics
    echo -e "${YELLOW}=== Stream Metrics ===${NC}"
    EXEC_COUNT=$(redis-cli XLEN quantum:stream:execution.result 2>/dev/null)
    HARVEST_COUNT=$(redis-cli XLEN quantum:stream:harvest.suggestions 2>/dev/null)
    INTENT_COUNT=$(redis-cli XLEN quantum:stream:trade.intent 2>/dev/null)
    
    echo "  execution.result: $EXEC_COUNT entries"
    echo "  harvest.suggestions: $HARVEST_COUNT entries"
    echo "  trade.intent: $INTENT_COUNT entries"
    echo ""
    
    # Consumer Group
    echo -e "${YELLOW}=== Consumer Group ===${NC}"
    CONSUMER_INFO=$(redis-cli XINFO GROUPS quantum:stream:execution.result 2>/dev/null | grep -A 6 "harvest_brain:execution")
    CONSUMERS=$(echo "$CONSUMER_INFO" | grep "^consumers$" -A 1 | tail -1)
    PENDING=$(echo "$CONSUMER_INFO" | grep "^pending$" -A 1 | tail -1)
    LAG=$(echo "$CONSUMER_INFO" | grep "^lag$" -A 1 | tail -1)
    
    echo "  Group: harvest_brain:execution"
    echo "  Consumers: $CONSUMERS"
    echo "  Pending: $PENDING"
    
    if [[ "$LAG" == "0" ]]; then
        echo -e "  Lag: ${GREEN}$LAG${NC} (no backlog)"
    elif [[ "$LAG" -lt "100" ]]; then
        echo -e "  Lag: ${YELLOW}$LAG${NC} (minor backlog)"
    else
        echo -e "  Lag: ${RED}$LAG${NC} (significant backlog)"
    fi
    echo ""
    
    # Dedup Keys
    echo -e "${YELLOW}=== Dedup Status ===${NC}"
    DEDUP_COUNT=$(redis-cli KEYS "quantum:dedup:harvest:*" 2>/dev/null | wc -l)
    echo "  Active dedup keys: $DEDUP_COUNT"
    echo ""
    
    # Recent Harvest Suggestions
    echo -e "${YELLOW}=== Recent Harvest Suggestions ===${NC}"
    if [[ "$HARVEST_COUNT" -gt "0" ]]; then
        redis-cli XREVRANGE quantum:stream:harvest.suggestions + - COUNT 3 2>/dev/null | head -20
    else
        echo "  No harvest suggestions yet"
    fi
    echo ""
    
    # Recent Errors
    echo -e "${YELLOW}=== Recent Errors ===${NC}"
    ERRORS=$(tail -100 /var/log/quantum/harvest_brain.log 2>/dev/null | grep -i ERROR | tail -3)
    if [[ -z "$ERRORS" ]]; then
        echo -e "  ${GREEN}No recent errors${NC}"
    else
        echo -e "${RED}$ERRORS${NC}"
    fi
    echo ""
    
    # Footer
    echo -e "${BLUE}════════════════════════════════════════════════════════════════════${NC}"
    echo "Refreshing every 5 seconds... (Ctrl+C to exit)"
    echo "Log file: tail -f /var/log/quantum/harvest_brain.log"
    echo ""
    
    sleep 5
done
