import subprocess, os, sys

SSH_KEY = os.path.expanduser('~/.ssh/hetzner_fresh')
VPS = 'root@46.224.116.254'

# 1. Deploy updated shadow controller
local = '/mnt/c/quantum_trader/microservices/shadow_mode_controller/shadow_mode_controller.py'
remote = '/opt/quantum/microservices/shadow_mode_controller/shadow_mode_controller.py'
with open(local, 'rb') as f:
    data = f.read()
clean = data.replace(b'\r\n', b'\n').replace(b'\r', b'')
r = subprocess.run(['ssh', '-i', SSH_KEY, VPS, f'cat > {remote}'], input=clean, capture_output=True)
print(f'Deploy shadow controller: {"OK" if r.returncode == 0 else "ERR: " + r.stderr.decode()}')

# 2. Destroy old consumer group (forces full replay from id=0 on next start)
cmd = ['ssh', '-i', SSH_KEY, VPS,
       'redis-cli XGROUP DESTROY quantum:stream:harvest.v2.shadow shadow_controller']
r2 = subprocess.run(cmd, capture_output=True, text=True)
print(f'XGROUP DESTROY: {r2.stdout.strip()} (1=destroyed, 0=did not exist)')

# 3. Initialize quantum:dag8:current_phase = 0 if missing
cmd3 = ['ssh', '-i', SSH_KEY, VPS,
        'redis-cli EXISTS quantum:dag8:current_phase']
r3 = subprocess.run(cmd3, capture_output=True, text=True)
if r3.stdout.strip() == '0':
    r4 = subprocess.run(['ssh', '-i', SSH_KEY, VPS,
                         'redis-cli SET quantum:dag8:current_phase 0'],
                        capture_output=True, text=True)
    print(f'SET current_phase=0: {r4.stdout.strip()}')
else:
    print(f'current_phase already set (value kept)')

# 4. Restart shadow controller
r5 = subprocess.run(['ssh', '-i', SSH_KEY, VPS,
                     'systemctl restart quantum-shadow-mode-controller'],
                    capture_output=True)
print(f'Restart shadow controller: {"OK" if r5.returncode == 0 else "ERR"}')

print('=== SHADOW FIX DEPLOYED ===')
print('Will replay 651 stream messages. Check in 60s:')
print('  redis-cli hgetall quantum:sandbox:accuracy:latest')
