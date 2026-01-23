#!/usr/bin/env bash
set -euo pipefail

SYMS=("BTCUSDT" "ETHUSDT" "SOLUSDT")

ok(){ echo "✅ $*"; }
warn(){ echo "⚠️  $*"; }
fail(){ echo "❌ $*"; }

echo "=== P0→P1.5 LOCK PROOF PACK ==="
date -u
echo

echo "== P0: Core health =="
redis-cli PING >/dev/null && ok "Redis PING" || fail "Redis PING failed"
systemctl is-active quantum-harvest-proposal >/dev/null && ok "quantum-harvest-proposal active" || warn "quantum-harvest-proposal not active"
systemctl is-active quantum-harvest-metrics-exporter >/dev/null && ok "harvest exporter active" || warn "harvest exporter not active"
echo

echo "== P0.5: MarketState keys exist + have fields =="
for s in "${SYMS[@]}"; do
  k="quantum:marketstate:$s"
  if [[ "$(redis-cli EXISTS "$k")" != "1" ]]; then
    fail "$k missing"
    continue
  fi
  cnt="$(redis-cli HLEN "$k")"
  [[ "$cnt" -gt 5 ]] && ok "$k HLEN=$cnt" || warn "$k suspiciously small HLEN=$cnt"
  redis-cli HGETALL "$k" | head -20
  echo
done

echo "== P1.5: Risk Proposal keys exist + have fields =="
for s in "${SYMS[@]}"; do
  k="quantum:risk:proposal:$s"
  if [[ "$(redis-cli EXISTS "$k")" != "1" ]]; then
    warn "$k missing"
    continue
  fi
  cnt="$(redis-cli HLEN "$k")"
  [[ "$cnt" -gt 5 ]] && ok "$k HLEN=$cnt" || warn "$k suspiciously small HLEN=$cnt"
  redis-cli HGETALL "$k" | head -25
  echo
done

echo "== P2.x sanity: Harvest Proposal has K components + timestamp freshness =="
now="$(date +%s)"
for s in "${SYMS[@]}"; do
  k="quantum:harvest:proposal:$s"
  if [[ "$(redis-cli EXISTS "$k")" != "1" ]]; then
    warn "$k missing"
    continue
  fi
  # Required fields from P2.6B + P2.7C.1
  vals="$(redis-cli HMGET "$k" kill_score k_regime_flip k_sigma_spike k_ts_drop k_age_penalty last_update_epoch harvest_action R_net new_sl_proposed)"
  echo "--- $s ---"
  echo "$vals"
  # freshness
  ts="$(redis-cli HGET "$k" last_update_epoch || true)"
  if [[ "$ts" =~ ^[0-9]+$ ]]; then
    age=$(( now - ts ))
    if [[ "$age" -lt 30 ]]; then ok "$s last_update_epoch age=${age}s (<30s)"; else warn "$s last_update_epoch age=${age}s (>=30s)"; fi
  else
    warn "$s last_update_epoch missing/invalid: '$ts'"
  fi
  echo
done

echo "== Prometheus: can we see harvest metrics? =="
if curl -s http://127.0.0.1:9091/-/ready >/dev/null 2>&1; then
  ok "Prometheus ready (:9091)"
  # Query and parse in one go
  python3 <<'PY'
import json, urllib.request
try:
    url = "http://127.0.0.1:9091/api/v1/query?query=quantum_harvest_kill_score"
    with urllib.request.urlopen(url) as resp:
        d = json.loads(resp.read())
    res = d.get("data", {}).get("result", [])
    print(f"symbols={len(res)}")
    for r in res:
        sym = r["metric"].get("symbol")
        val = r["value"][1]
        print(f"  {sym}: {val}")
except Exception as e:
    print(f"⚠️  Error querying Prometheus: {e}")
PY
else
  warn "Prometheus not reachable on :9091"
fi

echo
echo "=== DONE ==="
