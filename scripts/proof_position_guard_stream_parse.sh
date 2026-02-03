#!/bin/bash
# Position Guard Stream Parse (Field-Aware Parsing)
# Handles Redis stream XREVRANGE output correctly using awk

set -e

echo "=================================================="
echo "POSITION GUARD STREAM VERIFICATION (Field-Aware)"
echo "=================================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "Parsing last 300 apply.plan entries from Redis stream..."
echo ""

# Field-aware parsing with awk
result=$(redis-cli --raw XREVRANGE quantum:stream:apply.plan + - COUNT 300 2>/dev/null | awk '
BEGIN {
    sym=""
    dec=""
    rc=""
    action=""
    btc_skip=0
    eth_skip=0
    btc_total=0
    eth_total=0
}

# Track current entry ID
/^[0-9]+-[0-9]+$/ {
    # Process previous entry before starting new one
    if (sym != "") {
        if (sym ~ /BTCUSDT/) {
            btc_total++
            if (dec == "SKIP" && rc ~ /update_sl_no_position_skip/)
                btc_skip++
        }
        if (sym ~ /ETHUSDT/) {
            eth_total++
            if (dec == "SKIP" && rc ~ /update_sl_no_position_skip/)
                eth_skip++
        }
    }
    # Reset for next entry
    sym=""
    dec=""
    rc=""
    action=""
    next
}

# Capture field names and values
NR % 2 == 0 {
    field = prev_line
    value = $0
    
    if (field == "symbol")
        sym = value
    else if (field == "decision")
        dec = value
    else if (field == "reason_codes")
        rc = value
    else if (field == "action")
        action = value
}

# Store previous line for field name
{
    prev_line = $0
}

END {
    # Process last entry
    if (sym != "") {
        if (sym ~ /BTCUSDT/) {
            btc_total++
            if (dec == "SKIP" && rc ~ /update_sl_no_position_skip/)
                btc_skip++
        }
        if (sym ~ /ETHUSDT/) {
            eth_total++
            if (dec == "SKIP" && rc ~ /update_sl_no_position_skip/)
                eth_skip++
        }
    }
    
    print "BTCUSDT_TOTAL=" btc_total
    print "BTCUSDT_SKIP=" btc_skip
    print "ETHUSDT_TOTAL=" eth_total
    print "ETHUSDT_SKIP=" eth_skip
}
')

# Parse awk output
btc_total=$(echo "$result" | grep "BTCUSDT_TOTAL=" | cut -d= -f2)
btc_skip=$(echo "$result" | grep "BTCUSDT_SKIP=" | cut -d= -f2)
eth_total=$(echo "$result" | grep "ETHUSDT_TOTAL=" | cut -d= -f2)
eth_skip=$(echo "$result" | grep "ETHUSDT_SKIP=" | cut -d= -f2)

echo "Stream Analysis Results:"
echo "  BTCUSDT: $btc_total total plans, $btc_skip with position guard"
echo "  ETHUSDT: $eth_total total plans, $eth_skip with position guard"
echo ""

# Test 1: BTC/ETH have position guard evidence
if [ "$btc_skip" -gt 0 ] || [ "$eth_skip" -gt 0 ]; then
    echo -e "${GREEN}✓ TEST 1: Position guard active in stream${NC}"
    echo "  Evidence: BTC+ETH skips = $((btc_skip + eth_skip))"
else
    echo -e "${RED}✗ TEST 1: No position guard evidence in stream${NC}"
fi
echo ""

# Test 2: Get sample BTC plan with reason_codes
echo "Sample BTC Plan (showing three-layer chain):"
echo ""
redis-cli --raw XREVRANGE quantum:stream:apply.plan + - COUNT 300 2>/dev/null | awk '
BEGIN {
    in_btc=0
    field=""
}

/^[0-9]+-[0-9]+$/ {
    if (in_btc) exit
    in_btc=0
    entry_id=$0
}

NR % 2 == 0 {
    field = prev_line
    value = $0
    
    if (field == "symbol" && value == "BTCUSDT") {
        in_btc=1
        print "  symbol: " value
    }
    else if (in_btc) {
        if (field == "decision" || field == "reason_codes" || field == "action" || field == "steps")
            print "  " field ": " value
    }
}

{
    prev_line = $0
}
' | head -10

echo ""

# Test 3: Verify three-layer chain exists
chain_check=$(redis-cli --raw XREVRANGE quantum:stream:apply.plan + - COUNT 300 2>/dev/null | awk '
BEGIN {
    sym=""
    rc=""
    chain_found=0
}

/^[0-9]+-[0-9]+$/ {
    if (sym != "" && sym ~ /BTCUSDT|ETHUSDT/) {
        if (rc ~ /kill_score_open_ok/ && rc ~ /update_sl_no_position_skip/ && rc ~ /action_hold/)
            chain_found=1
    }
    sym=""
    rc=""
    next
}

NR % 2 == 0 {
    field = prev_line
    value = $0
    
    if (field == "symbol")
        sym = value
    else if (field == "reason_codes")
        rc = value
}

{
    prev_line = $0
}

END {
    if (chain_found)
        print "FOUND"
    else
        print "NOT_FOUND"
}
')

if [ "$chain_check" == "FOUND" ]; then
    echo -e "${GREEN}✓ TEST 2: Three-layer contract chain verified${NC}"
    echo "  Layers: kill_score_open_ok → update_sl_no_position_skip → action_hold"
else
    echo -e "${YELLOW}⚠ TEST 2: Three-layer chain not detected${NC}"
fi
echo ""

# Summary
echo "=================================================="
echo "SUMMARY"
echo "=================================================="
echo ""

if [ "$btc_skip" -gt 0 ] || [ "$eth_skip" -gt 0 ]; then
    echo -e "${GREEN}Position guard IS ACTIVE in stream${NC}"
    echo "  BTC/ETH plans with guard: $((btc_skip + eth_skip))"
    echo "  Stream parsing: Field-aware (robust)"
    exit 0
else
    echo -e "${RED}Position guard NOT DETECTED in stream${NC}"
    echo "  This may indicate:"
    echo "    - No recent BTC/ETH proposals with new_sl_proposed"
    echo "    - Guard not deployed yet"
    echo "    - Stream analysis window too small"
    exit 1
fi
