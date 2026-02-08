#!/bin/bash
cd /root/quantum_trader
mkdir -p /root/logs
python3 post_calibration_monitor.py > /root/logs/monitor_console.log 2>&1 &
echo $! > /root/logs/monitor.pid
echo "✅ Monitor started (PID: $(cat /root/logs/monitor.pid))"
