#!/bin/bash
echo "=== position_state_brain: permit creation code ==="
grep -n "permit:p33\|permit_key\|r\.set\|redis\.set\|r\.hset\|redis\.hset" \
    /opt/quantum/microservices/position_state_brain/main.py 2>/dev/null | head -40

echo ""
echo "=== apply_layer: permit creation code (lines p33_key) ==="
grep -n "p33_key\|permit:p33\|hset.*permit\|set.*permit" \
    /opt/quantum/microservices/apply_layer/main.py 2>/dev/null | grep -v "bak\|#" | head -20

echo ""
echo "=== portfolio_gate: permit creation ==="
grep -n "permit:p33\|permit_key.*set\|r\.set\|hset.*permit" \
    /opt/quantum/microservices/portfolio_gate/main.py 2>/dev/null | head -20

echo ""
echo "=== position_state_brain lines 1-50 (import/description) ==="
head -50 /opt/quantum/microservices/position_state_brain/main.py

echo ""
echo "=== Current STRING vs HASH permits ==="
TOTAL=$(redis-cli KEYS "quantum:permit:p33:*" 2>/dev/null | wc -l)
STRING_C=0
HASH_C=0
for k in $(redis-cli KEYS "quantum:permit:p33:*" 2>/dev/null | head -20); do
    T=$(redis-cli TYPE "$k")
    if [ "$T" = "string" ]; then
        STRING_C=$((STRING_C+1))
        echo "  STRING: $k = $(redis-cli GET $k 2>/dev/null)"
    else
        HASH_C=$((HASH_C+1))
    fi
done
echo "  Total: $TOTAL | STRING: $STRING_C | HASH: $HASH_C"
