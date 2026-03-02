#!/bin/bash
echo "=== intent_executor lines 1030-1100 ==="
sed -n '1030,1100p' /opt/quantum/microservices/intent_executor/main.py

echo ""
echo "=== Search: how is last_exec_ts read (cooldown check) ==="
grep -n "last_exec_ts\|exec_cooldown\|cooldown_sec\|cooldown_ms\|COOLDOWN" /opt/quantum/microservices/intent_executor/main.py | head -30

echo ""
echo "=== governor: cooldown references ==="
grep -rn "last_exec_ts\|cooldown" /opt/quantum/microservices/portfolio_risk_governor/main.py 2>/dev/null | head -20
