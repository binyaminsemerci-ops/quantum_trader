#!/usr/bin/env bash
set -euo pipefail

symbols=(BTCUSDT ETHUSDT BNBUSDT SOLUSDT)
for sym in "${symbols[@]}"; do
  ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  case "$sym" in
    BTCUSDT) u=120.5; pct=0.42;;
    ETHUSDT) u=85.3; pct=0.37;;
    BNBUSDT) u=-25.7; pct=-0.11;;
    SOLUSDT) u=44.1; pct=0.18;;
  esac
  json=$(printf '{"symbol":"%s","unrealized_pnl":%.4f,"unrealized_pct":%.4f,"realized_pnl":0,"realized_pct":0,"total_pnl":%.4f,"realized_trades":3,"position_size":1.0,"side":"long","entry_price":0,"notional":0,"leverage":3,"timestamp":"%s","source":"manual_seed"}' "$sym" "$u" "$pct" "$u" "$ts")
  redis-cli SETEX "quantum:rl:reward:${sym}" 3600 "$json" >/dev/null
done
curl -s https://app.quantumfond.com/api/rl-dashboard/
