#!/bin/bash
while true; do
  cd /home/qt/quantum_trader
  python3 /tmp/_monitor.py >> /tmp/qt_monitor.log 2>&1
  sleep 600
done
