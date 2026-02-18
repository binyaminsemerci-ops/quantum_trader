import re
from pathlib import Path
from datetime import datetime, timezone

path = Path('c:/quantum_trader/tmp/truthmap_probe.out')
text = path.read_text(encoding='utf-8', errors='ignore').splitlines()

stream_re = re.compile(r'^--- (quantum:stream:[^\s]+)')

streams = {}
current = None
buffer = []

def finalize(name, lines):
    if not name:
        return
    info = {
        'length': None,
        'last_entry': None,
        'last_timestamp': None,
        'source': None,
        'groups': [],
    }
    for i, line in enumerate(lines):
        if line == 'length' and i + 1 < len(lines):
            info['length'] = lines[i + 1]
            break

    for i, line in enumerate(lines):
        if line == 'name' and i + 1 < len(lines):
            info['groups'].append(lines[i + 1])

    for i, line in enumerate(lines):
        if line == 'last-entry' and i + 1 < len(lines):
            info['last_entry'] = lines[i + 1]
            for j in range(i + 2, min(i + 80, len(lines))):
                if lines[j] == 'timestamp' and j + 1 < len(lines):
                    info['last_timestamp'] = lines[j + 1]
                if lines[j] == 'ts_epoch' and j + 1 < len(lines) and not info['last_timestamp']:
                    try:
                        ts = int(lines[j + 1])
                        info['last_timestamp'] = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                    except Exception:
                        pass
                if lines[j] == 'source' and j + 1 < len(lines):
                    info['source'] = lines[j + 1]
            break

    streams[name] = info

for line in text:
    m = stream_re.match(line)
    if m:
        finalize(current, buffer)
        current = m.group(1)
        buffer = []
    else:
        if current is not None:
            buffer.append(line)

finalize(current, buffer)

out = Path('c:/quantum_trader/tmp/truthmap_stream_table.md')
with out.open('w', encoding='utf-8') as f:
    f.write('| STREAM | PRODUSER (source) | CONSUMER GROUPS | AKTIV? | SIST EVENT | LENGTH |\n')
    f.write('|---|---|---|---|---|---|\n')
    for name in sorted(streams.keys()):
        info = streams[name]
        length = info['length'] or 'UNKNOWN'
        active = 'YES' if info['length'] and info['length'] != '0' else 'NO'
        producer = info['source'] or 'UNKNOWN'
        consumers = ','.join(info['groups']) if info['groups'] else 'NONE'
        last_event = info['last_timestamp'] or info['last_entry'] or 'UNKNOWN'
        f.write(f'| {name} | {producer} | {consumers} | {active} | {last_event} | {length} |\n')

print(f'Wrote {out}')
