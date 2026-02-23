#!/bin/bash
echo "=== apply.plan siste 20 per symbol ==="
redis-cli xrevrange quantum:stream:apply.plan + - COUNT 200 2>/dev/null | python3 -c "
import sys, re
entries = []
cid, cf, fn = None, {}, None  
for l in sys.stdin:
    l = l.strip()
    if not l: continue
    if re.match(r'^\d+-\d+$', l):
        if cid: entries.append(cf)
        cid, cf, fn = l, {'_id': l}, None
    elif fn is None: fn = l
    else: cf[fn] = l; fn = None
if cid: entries.append(cf)
from collections import Counter
syms = Counter(e.get('symbol','?') + ':' + e.get('action','?') for e in entries)
for k, v in syms.most_common(20):
    print(f'  {v:3d}x {k}')
print(f'  Total: {len(entries)} entries')
"

echo ""
echo "=== Exitbrain — hva publiserer den? ==="
grep -n "xadd\|apply.plan\|quantum:stream" /home/qt/quantum_trader/microservices/exitbrain_v3_5/service.py 2>/dev/null | head -20

echo ""
echo "=== AAVEUSDT i allowlist? ==="
redis-cli hgetall quantum:apply_layer:allowlist 2>/dev/null | head -20
redis-cli get quantum:apply_layer:allowlist 2>/dev/null
redis-cli smembers quantum:allowlist:symbols 2>/dev/null | head -30

echo ""
echo "=== apply_layer env - ALLOWLIST ==="
grep -i allowlist /etc/quantum/apply-layer.env 2>/dev/null | head -5

echo ""
echo "=== exitbrain siste logs ==="
journalctl -u quantum-exitbrain-v3-5 --no-pager --since "3 minutes ago" -q 2>/dev/null | tail -15
# Try alternate naming
for svc in quantum-exit-intelligence quantum-exitbrain; do
  journalctl -u $svc --no-pager --since "3 minutes ago" -q 2>/dev/null | grep -E "AAVEUSDT|ACEUSDT|plan\|close" | tail -5
done
