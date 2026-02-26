#!/bin/bash

# ── 1. Clear phantom symbols from harvest-proposal ──────────────
echo "=== FIX 1: Empty SYMBOLS in harvest-proposal.env ==="
sed -i 's/^SYMBOLS=.*/SYMBOLS=/' /etc/quantum/harvest-proposal.env
grep "^SYMBOLS" /etc/quantum/harvest-proposal.env
systemctl restart quantum-harvest-proposal.service
sleep 3
systemctl is-active quantum-harvest-proposal.service && echo "  service: OK" || echo "  service: FAILED"

echo "  Deleting phantom harvest proposals..."
redis-cli keys "quantum:harvest:proposal:*" | xargs -r redis-cli del
echo "  Done"

# ── 2. Delete entry=0.0 phantom ledger positions ─────────────────
echo ""
echo "=== FIX 3: Delete entry=0.0 phantom ledger keys ==="
for key in quantum:position:ledger:BTCUSDT quantum:position:ledger:BNBUSDT quantum:position:ledger:SOLUSDT; do
  result=$(redis-cli del "$key")
  echo "  del $key -> $result"
done

# ── 3. Discover real portfolio/balance state keys ─────────────────
echo ""
echo "=== DIAG: Real portfolio/balance keys ==="
echo "--- quantum:state:* ---"
redis-cli keys "quantum:state:*" | head -20
echo "--- quantum:balance:* ---"
redis-cli keys "quantum:balance:*" | head -10
echo "--- quantum:state:portfolio type ---"
redis-cli type "quantum:state:portfolio"
echo "--- quantum:state:portfolio value (first 500 chars) ---"
redis-cli get "quantum:state:portfolio" | head -c 500
echo ""
echo "--- portfolio-state-publisher last log ---"
journalctl -u quantum-portfolio-state-publisher -n 5 --no-pager | tail -5

echo ""
echo "=== DONE ==="
