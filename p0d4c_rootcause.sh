#!/bin/bash
set -euo pipefail
TS=$(date -u +%Y%m%d_%H%M%S)
R=/tmp/p0d4c_exec_rootcause_$TS.txt
echo "=== P0.D.4c EXECUTION ROOT-CAUSE MAP ===" | tee $R
echo "Time: $(date -u '+%Y-%m-%d %H:%M:%S') UTC" | tee -a $R
echo "Host: $(hostname)" | tee -a $R
echo | tee -a $R

echo "=== 1) WHAT LISTENS ON :8002 ===" | tee -a $R
ss -ltnp | grep -E ":8002\b" | tee -a $R || echo "No process on :8002" | tee -a $R
echo | tee -a $R

PID=$(ss -ltnp | awk '/:8002/ {print $NF}' | head -1 | sed -n 's/.*pid=\([0-9]\+\).*/\1/p')
echo "PID: $PID" | tee -a $R
echo | tee -a $R

if [ -n "$PID" ]; then
  echo "=== 2) PROCESS IDENTITY ===" | tee -a $R
  ps -fp $PID | tee -a $R
  echo "cmdline:" | tee -a $R
  tr "\0" " " < /proc/$PID/cmdline | tee -a $R
  echo | tee -a $R
  echo | tee -a $R

  echo "cwd:" | tee -a $R
  readlink -f /proc/$PID/cwd | tee -a $R
  echo "exe:" | tee -a $R
  readlink -f /proc/$PID/exe | tee -a $R
  echo | tee -a $R

  echo "=== 3) ENV HINTS (filtered) ===" | tee -a $R
  tr "\0" "\n" < /proc/$PID/environ | egrep -i "(QUANTUM|REDIS|STREAM|EXEC|RESULT|BRIDGE|GROUP|ENV|MODE|BINANCE|TESTNET)" | sort | tee -a $R || echo "No matching env vars" | tee -a $R
  echo | tee -a $R
fi

echo "=== 4) SYSTEMD: WHAT UNITS EXIST AROUND EXECUTION ===" | tee -a $R
systemctl list-unit-files --no-legend | egrep -i "quantum.*(exec|bridge|trade|bot|risk|ai)" | awk '{print $1, $2}' | sort | tee -a $R || echo "No matching units" | tee -a $R
echo | tee -a $R

echo "=== 5) WHICH UNIT OWNS PID (if any) ===" | tee -a $R
if [ -n "$PID" ]; then
  systemctl status $PID --no-pager -l 2>&1 | head -80 | tee -a $R || echo "PID not owned by systemd" | tee -a $R
fi
echo | tee -a $R

echo "=== 6) REDIS KEYS: execution result streams ===" | tee -a $R
redis-cli --scan --pattern "quantum:stream:*execution*" | sort | tee -a $R
echo | tee -a $R

echo "REPORT SAVED: $R"
echo | tee -a $R
cat $R
