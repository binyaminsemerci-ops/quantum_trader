#!/bin/bash
SERVICES=(
  "quantum-execution-result-bridge"
  "quantum-rl-trainer"
  "quantum-portfolio-state-publisher"
  "quantum-balance-tracker"
  "quantum-policy-sync.timer"
  "verify-rl.timer"
  "quantum-rl-shadow-metrics-exporter"
  "quantum-rl-shadow-scorecard.timer"
  "quantum-trade-logger"
  "quantum-stream-recover.timer"
  "quantum-performance-tracker"
  "quantum-strategic-memory"
)

echo "====== SERVICE STATUS PROOF $(date -u '+%Y-%m-%d %H:%M:%S UTC') ======"
for svc in "${SERVICES[@]}"; do
  state=$(systemctl is-active "$svc" 2>/dev/null)
  enabled=$(systemctl is-enabled "$svc" 2>/dev/null)
  echo "[$state] $svc ($enabled)"
done
echo "====== END ======"
