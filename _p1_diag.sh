#!/bin/bash
echo "=== P1 DIAGNOSE: MANUAL_LANE_OFF ==="
echo ""
echo "--- Alle lane-relaterte Redis-nøkler ---"
redis-cli KEYS 'quantum:*lane*' 2>/dev/null
redis-cli KEYS 'quantum:*manual*' 2>/dev/null
redis-cli KEYS 'quantum:*LANE*' 2>/dev/null
redis-cli KEYS 'quantum:*MANUAL*' 2>/dev/null

echo ""
echo "--- Intent-executor kildekode: søk etter MANUAL_LANE_OFF ---"
grep -rn "MANUAL_LANE_OFF\|manual_lane\|MANUAL_LANE\|lane_off\|lane_manual" \
    /opt/quantum/microservices/intent_executor/main.py 2>/dev/null | head -30

echo ""
echo "--- Intent-executor kildekode: søk etter lane-nøkler ---"
grep -rn "quantum:lane\|quantum:manual\|lane_switch\|lane_key" \
    /opt/quantum/microservices/intent_executor/main.py 2>/dev/null | head -20

echo ""
echo "--- Intent-executor full MANUAL_LANE kontekst (±10 linjer) ---"
LINE=$(grep -n "MANUAL_LANE_OFF" /opt/quantum/microservices/intent_executor/main.py 2>/dev/null | head -1 | cut -d: -f1)
if [ -n "$LINE" ]; then
    sed -n "$((LINE-10)),$((LINE+10))p" /opt/quantum/microservices/intent_executor/main.py
else
    echo "Ikke funnet i main.py, søker i andre filer..."
    grep -rn "MANUAL_LANE_OFF" /opt/quantum/microservices/ 2>/dev/null | head -10
fi

echo ""
echo "--- Intent-executor logg siste 30 linjer ---"
journalctl -u quantum-intent-executor.service -n 30 --no-pager 2>/dev/null | tail -30
