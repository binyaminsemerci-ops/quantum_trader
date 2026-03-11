#!/bin/bash
# start_injection.sh — Start scenario injection in background
cd /home/qt/quantum_trader
nohup /home/qt/quantum_trader_venv/bin/python _inject_replay_scenarios.py > /tmp/scenario_injection.log 2>&1 &
echo "PID: $!"
echo "Log: /tmp/scenario_injection.log"
tail -1 /tmp/scenario_injection.log 2>/dev/null || echo "(log not yet written)"
