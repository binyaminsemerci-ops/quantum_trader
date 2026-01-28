from pathlib import Path
path = Path('/etc/systemd/system/quantum-ai-strategy-router.service')
data = path.read_text()
old = '/usr/local/bin/ai_strategy_router.py'
new = '/home/qt/quantum_trader/ai_strategy_router.py'
if old in data:
    path.write_text(data.replace(old, new))
    print('Updated ExecStart to repo path')
else:
    print('ExecStart already set or pattern missing')
