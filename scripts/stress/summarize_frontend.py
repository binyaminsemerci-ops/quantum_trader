import json
from pathlib import Path
p = Path('artifacts/stress/frontend_aggregated.json')
if not p.exists():
    print('missing', p)
    raise SystemExit(1)
j = json.loads(p.read_text())
ok = err = skipped = other = 0
durs = []
for r in j['runs']:
    s = r.get('summary')
    if s == 0:
        ok += 1
    elif s == 'error':
        err += 1
    elif s == 'skipped':
        skipped += 1
    else:
        other += 1
    d = r.get('duration')
    if isinstance(d, (int, float)):
        durs.append(d)
print('total', len(j['runs']))
print('ok', ok, 'error', err, 'skipped', skipped, 'other', other)
print('avg_duration', sum(durs) / len(durs) if durs else None)
