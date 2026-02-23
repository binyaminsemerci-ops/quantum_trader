#!/bin/bash
echo "=== Consumer group pending messages ==="
redis-cli xpending quantum:stream:apply.plan apply_layer_entry - + 20 2>/dev/null

echo ""
echo "=== Siste 10 apply.result entries ==="
redis-cli xrevrange quantum:stream:apply.result + - COUNT 10 2>/dev/null | python3 -c "
import sys, re
entries = []
cid, cf, fn = None, {}, None
for l in sys.stdin:
    l = l.strip()
    if not l: continue
    if re.match(r'^\d+-\d+$', l):
        if cid: entries.append(cf)
        cid, cf, fn = l, {}, None
    elif fn is None: fn = l
    else: cf[fn] = l; fn = None
if cid: entries.append(cf)
for e in entries:
    print(f\"  {e.get('symbol','?')} {e.get('action','?')} executed={e.get('executed','?')} error={e.get('error','?')}\")
"

echo ""
echo "=== Active position count (correct filter) ==="
redis-cli keys "quantum:position:*" 2>/dev/null | python3 -c "
import sys
keys = [l.strip() for l in sys.stdin if l.strip()]
active = [k for k in keys if 'snapshot' not in k and 'ledger' not in k and 'cooldown' not in k]
print(f'Total keys: {len(keys)}')
print(f'Active (no snapshot/ledger/cooldown): {len(active)}')
for k in sorted(active):
    print(f'  {k}')
"

echo ""
echo "=== apply_layer ENTRY logs siste 5 min ==="
journalctl -u quantum-apply-layer --no-pager --since "5 minutes ago" -q 2>/dev/null | grep "ENTRY\|entry\|AAVEUSDT\|ACEUSDT" | tail -20

echo ""
echo "=== Binance testnet credentials satt? ==="
grep -l BINANCE_TESTNET_API_KEY /etc/quantum/*.env 2>/dev/null | head -3
grep "BINANCE_TESTNET_API_KEY" /etc/quantum/apply-layer.env 2>/dev/null | head -2
