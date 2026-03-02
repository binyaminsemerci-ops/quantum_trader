#!/bin/bash
echo "=== harvest-v2 service unit ==="
cat /etc/systemd/system/quantum-harvest-v2.service | grep -E "ExecStart|WorkingDir|Environment|User"

echo ""
echo "=== intent-executor service unit ==="
cat /etc/systemd/system/quantum-intent-executor.service | grep -E "ExecStart|WorkingDir|Environment|User"

echo ""
echo "=== which position_provider.py is used by running harvest-v2? ==="
PID=$(systemctl show quantum-harvest-v2.service -p MainPID | cut -d= -f2)
echo "  PID: $PID"
ls -la /proc/$PID/cwd 2>/dev/null
cat /proc/$PID/cmdline 2>/dev/null | tr '\0' ' '
echo ""

echo ""
echo "=== Find ALL position_provider.py files ==="
find / -name "position_provider.py" 2>/dev/null | grep -v __pycache__

echo ""
echo "=== ADAUSDT cooldown details ==="
redis-cli TTL "quantum:cooldown:last_exec_ts:ADAUSDT"
redis-cli GET "quantum:cooldown:last_exec_ts:ADAUSDT"

echo ""
echo "=== Intent-executor logs (last 2 min) ==="
journalctl -u quantum-intent-executor.service --since "2 minutes ago" --no-pager 2>/dev/null | grep -E "ADA|ADAUSDT|cooldown|Executing|FILLED" | tail -20
