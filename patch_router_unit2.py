from pathlib import Path
path = Path('/etc/systemd/system/quantum-ai-strategy-router.service')
lines = path.read_text().splitlines()
new_exec = 'ExecStart=/opt/quantum/venvs/ai-engine/bin/python3 /home/qt/quantum_trader/ai_strategy_router.py'
new_path = 'Environment="PATH=/opt/quantum/venvs/ai-engine/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"'
out = []
for line in lines:
    if line.startswith('ExecStart='):
        out.append(new_exec)
    elif line.startswith('Environment="PATH='):
        out.append(new_path)
    else:
        out.append(line)
if not any(l.startswith('Environment="PATH=') for l in out):
    try:
        idx = out.index('[Service]') + 1
    except ValueError:
        idx = 0
    out.insert(idx, new_path)
path.write_text('\n'.join(out) + '\n')
print('router unit patched')
