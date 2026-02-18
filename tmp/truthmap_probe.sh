#!/usr/bin/env bash
set -euo pipefail

section() {
  printf "\n=== %s ===\n" "$1"
}

section "TIMESTAMP"
date -u +"%Y-%m-%dT%H:%M:%SZ"

section "OS"
uname -a
cat /etc/os-release 2>/dev/null || true
uptime

section "SYSTEMD_QUANTUM_UNITS"
systemctl list-units 'quantum-*' --all --no-pager

section "SYSTEMD_QUANTUM_UNIT_FILES"
systemctl list-unit-files 'quantum-*' --no-pager

section "SYSTEMD_FAILED"
systemctl --failed --no-pager

section "SYSTEMD_QUANTUM_DETAILS"
for unit in $(systemctl list-unit-files 'quantum-*' --no-legend | awk '{print $1}'); do
  echo "--- $unit"
  systemctl show "$unit" -p Id,ActiveState,SubState,Result,ExecMainStatus,ExecMainPID,ActiveEnterTimestamp,StateChangeTimestamp,Restart,RestartUSec
  echo
 done

section "PROCESS_QUANTUM"
ps -eo pid,ppid,cmd | grep -E 'quantum-|intent|apply|harvest|exit|ai-engine|trading_bot' | grep -v grep || true

section "ENV_FILES"
ls -la /etc/quantum || true

section "ENV_KEYS_REDACTED"
for f in /etc/quantum/*.env; do
  [ -e "$f" ] || continue
  echo "--- $f"
  awk -F= 'NF>=1 {key=$1; if (key ~ /(KEY|SECRET|PASS|TOKEN)/) {print key"=REDACTED"} else {print $0}}' "$f"
  echo
 done

section "REDIS_INFO"
redis-cli INFO server | egrep 'redis_version|uptime_in_seconds|role|connected_clients' || true

section "REDIS_STREAM_KEYS"
redis-cli --scan --pattern 'quantum:stream:*'

section "REDIS_STREAM_DETAILS"
for s in $(redis-cli --scan --pattern 'quantum:stream:*'); do
  echo "--- $s"
  redis-cli XINFO STREAM "$s"
  redis-cli XINFO GROUPS "$s" || true
  redis-cli XREVRANGE "$s" + - COUNT 1
  echo
 done

section "REDIS_KEY_SAMPLES"
redis-cli --scan --pattern 'quantum:cfg:*' | head -n 50
redis-cli --scan --pattern 'quantum:policy:*' | head -n 50
