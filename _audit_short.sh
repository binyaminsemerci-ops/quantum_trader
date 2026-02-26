#!/bin/bash
echo "=== POSITIONS ==="
redis-cli --scan --pattern "quantum:position:*" | grep -v snapshot | grep -v backup | grep -v phantom
COUNT=$(redis-cli --scan --pattern "quantum:position:*" | grep -v snapshot | grep -v backup | grep -v phantom | wc -l)
echo "COUNT=$COUNT"

echo "=== MANUAL LANE ==="
echo "value=$(redis-cli get quantum:manual_lane:enabled)"
echo "ttl=$(redis-cli ttl quantum:manual_lane:enabled)"

echo "=== HARVEST SYMBOLS ==="
grep SYMBOLS /etc/quantum/harvest-proposal.env

echo "=== EXECUTION PIPELINE LAST 5 MIN ==="
python3 -c "
import redis, re, time; from collections import Counter
r = redis.Redis()
entries = r.xrange('quantum:stream:apply.result', min=str((int(time.time())-300)*1000))
dec = Counter(d.get(b'decision',b'?').decode() for _,d in entries)
errs = [(d.get(b'symbol',b'?').decode(), re.search(r'\"code\":\s*(-?[0-9]+)', d.get(b'error',b'').decode())) for _,d in entries if d.get(b'error',b'')]
binance_errs = [(s,m.group(1)) for s,m in errs if m]
print('Entries:', len(entries), '| Decisions:', dict(dec.most_common(4)))
print('Binance errors:', Counter(binance_errs) if binance_errs else 'ZERO')
"

echo "=== DEAD SERVICES ==="
journalctl -u quantum-execution-result-bridge -n 3 --no-pager 2>/dev/null
echo "---rl-trainer---"
journalctl -u quantum-rl-trainer -n 3 --no-pager 2>/dev/null

echo "=== P35 GUARD STATS ==="
python3 -c "
import redis; r = redis.Redis()
k = 'quantum:intent_executor:stats'
if r.exists(k):
    d = {a.decode():b.decode() for a,b in r.hgetall(k).items()}
    for key in ['p35_guard_blocked','executed_true','total_signals']:
        print(f'  {key}: {d.get(key,\"?\")}')
else:
    import subprocess
    o = subprocess.run(['journalctl','-u','quantum-intent-executor','-n','50','--no-pager'],capture_output=True,text=True)
    for l in o.stdout.splitlines():
        if 'guard' in l.lower() or 'blocked' in l.lower() or 'kelly' in l.lower():
            print(' ',l.strip()[:100])
            break
"
