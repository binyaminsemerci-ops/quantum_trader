#!/bin/bash
echo "=== 1. COST MODEL BRUKES? ==="
grep -rn "cost_model" /opt/quantum/microservices/ 2>/dev/null | grep -v ".pyc" | head -15

echo ""
echo "=== 2. EXIT LOGIC I AI ENGINE ==="
grep -n "SELL\|LONG\|SHORT\|direction\|signal_side" /opt/quantum/microservices/ai_engine/service.py 2>/dev/null | grep -v "#" | head -30

echo ""
echo "=== 3. FEE FILTER I AI ENGINE? ==="
grep -n "fee\|breakeven\|min_profit\|cost" /opt/quantum/microservices/ai_engine/service.py 2>/dev/null | head -20

echo ""
echo "=== 4. HVEM SENDER SELL-SIGNAL? ==="
grep -rn "trade.intent\|publish.*SELL\|send.*SELL\|side.*SELL" /opt/quantum/microservices/ 2>/dev/null | grep -v ".pyc" | grep -v "test" | head -20

echo ""
echo "=== 5. EXITBRAIN STRUKTUR ==="
ls /opt/quantum/microservices/ | grep -i exit
find /opt/quantum/microservices/ -name "*exit*" -o -name "*exitbrain*" 2>/dev/null | grep ".py$" | head -10

echo ""
echo "=== 6. GOVERNOR LOGIKK ==="
grep -n "governor\|Governor\|approve\|reject" /opt/quantum/microservices/apply_layer/main.py | head -20

echo ""
echo "=== 7. LAYER SEPARATION - HVEM LYTTER PÅ HVA? ==="
grep -rn "XREAD\|xread\|consumer.*group\|stream" /opt/quantum/microservices/ai_engine/service.py 2>/dev/null | grep -v "#" | head -10
grep -rn "apply.plan\|trade.intent" /opt/quantum/microservices/ 2>/dev/null | grep -v ".pyc" | grep "XREAD\|xread\|add\|publish" | head -15
