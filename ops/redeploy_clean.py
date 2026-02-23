"""
Re-deploy all Python files from stdin URLs to VPS, using Python to strip CRLF.
Run as: python3 - < this_file   (on local WSL with SSH access to VPS)
- Reads each file from Windows /mnt/c path
- Strips CRLF safely 
- Deploys to VPS via Python ssh subprocess
"""
import subprocess, os, sys

files = [
    ('/mnt/c/quantum_trader/microservices/layer4_portfolio_optimizer/layer4_portfolio_optimizer.py',
     '/opt/quantum/microservices/layer4_portfolio_optimizer/layer4_portfolio_optimizer.py'),
    ('/mnt/c/quantum_trader/microservices/apply_layer/main.py',
     '/opt/quantum/microservices/apply_layer/main.py'),
    ('/mnt/c/quantum_trader/microservices/paper_trade_controller/paper_trade_controller.py',
     '/opt/quantum/microservices/paper_trade_controller/paper_trade_controller.py'),
    ('/mnt/c/quantum_trader/microservices/layer2_signal_promoter/layer2_signal_promoter.py',
     '/opt/quantum/microservices/layer2_signal_promoter/layer2_signal_promoter.py'),
]

SSH_KEY = os.path.expanduser('~/.ssh/hetzner_fresh')
VPS = 'root@46.224.116.254'

for local, remote in files:
    # Read and strip CRLF
    with open(local, 'rb') as f:
        data = f.read()
    clean = data.replace(b'\r\n', b'\n').replace(b'\r', b'')
    
    # Write to VPS via ssh stdin redirection
    cmd = ['ssh', '-i', SSH_KEY, VPS, f'cat > {remote}']
    result = subprocess.run(cmd, input=clean, capture_output=True)
    if result.returncode == 0:
        print(f'OK: {remote}')
    else:
        print(f'ERR({result.returncode}): {remote} — {result.stderr.decode()}')
        sys.exit(1)

# Restart services
for svc in ['quantum-layer4-portfolio-optimizer', 'quantum-apply-layer', 
            'quantum-paper-trade-controller', 'quantum-layer2-signal-promoter']:
    cmd = ['ssh', '-i', SSH_KEY, VPS, f'systemctl restart {svc}']
    r = subprocess.run(cmd, capture_output=True)
    print(f'RESTARTED({r.returncode}): {svc}')

print('=== REDEPLOY DONE ===')
