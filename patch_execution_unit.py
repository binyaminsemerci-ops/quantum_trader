from pathlib import Path
path = Path('/etc/systemd/system/quantum-execution.service')
lines = path.read_text().splitlines()
new_path = 'Environment="PATH=/opt/quantum/venvs/ai-engine/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"'
out = []
inserted = False
for line in lines:
    if line.startswith('Environment="PATH='):
        out.append(new_path)
        inserted = True
    else:
        out.append(line)
if not inserted:
    # place after EnvironmentFile if present else after [Service]
    try:
        idx = out.index('[Service]') + 1
    except ValueError:
        idx = 0
    # find after EnvironmentFile lines
    for i, l in enumerate(out):
        if l.startswith('EnvironmentFile='):
            idx = i + 1
    out.insert(idx, new_path)
path.write_text('\n'.join(out) + '\n')
print('execution unit patched PATH')
