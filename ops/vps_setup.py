
# Generate service files directly on VPS — no CRLF issues
paper_svc = """[Unit]
Description=Quantum Paper Trade Controller Phase 2 TESTNET execution
After=network.target redis.service
Wants=redis.service

[Service]
User=root
WorkingDirectory=/opt/quantum/microservices/paper_trade_controller
ExecStart=/opt/quantum/venvs/ai-client-base/bin/python /opt/quantum/microservices/paper_trade_controller/paper_trade_controller.py
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
Environment=REDIS_HOST=localhost
Environment=BINANCE_TESTNET_API_KEY=
Environment=BINANCE_TESTNET_API_SECRET=
Environment=MAX_NOTIONAL_USDT=200.0
Environment=DEFAULT_SIZE_USDT=50.0
Environment=LEVERAGE=3

[Install]
WantedBy=multi-user.target
"""

promoter_svc = """[Unit]
Description=Quantum Layer 2 Signal Auto-Promoter
After=network.target redis.service
Wants=redis.service

[Service]
User=root
WorkingDirectory=/opt/quantum/microservices/layer2_signal_promoter
ExecStart=/opt/quantum/venvs/ai-client-base/bin/python /opt/quantum/microservices/layer2_signal_promoter/layer2_signal_promoter.py
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
Environment=REDIS_HOST=localhost
Environment=CHECK_INTERVAL=60
Environment=TOP_N_SYMBOLS=10
Environment=MIN_SHARPE=2.0
Environment=MIN_KELLY_EDGE=0.01

[Install]
WantedBy=multi-user.target
"""

import os, subprocess

with open('/etc/systemd/system/quantum-paper-trade-controller.service', 'w', newline='\n') as f:
    f.write(paper_svc.strip() + '\n')
print("OK: quantum-paper-trade-controller.service written")

with open('/etc/systemd/system/quantum-layer2-signal-promoter.service', 'w', newline='\n') as f:
    f.write(promoter_svc.strip() + '\n')
print("OK: quantum-layer2-signal-promoter.service written")

# Also fix CRLF in Python files
for path in [
    '/opt/quantum/microservices/layer4_portfolio_optimizer/layer4_portfolio_optimizer.py',
    '/opt/quantum/microservices/apply_layer/main.py',
    '/opt/quantum/microservices/paper_trade_controller/paper_trade_controller.py',
    '/opt/quantum/microservices/layer2_signal_promoter/layer2_signal_promoter.py',
]:
    if os.path.exists(path):
        with open(path, 'rb') as f:
            data = f.read()
        fixed = data.replace(b'\r\n', b'\n').replace(b'\r', b'')
        with open(path, 'wb') as f:
            f.write(fixed)
        print(f"OK LF: {path}")

subprocess.run(['systemctl', 'daemon-reload'], check=True)
print("DAEMON_RELOADED")
subprocess.run(['systemctl', 'enable', 'quantum-paper-trade-controller', 'quantum-layer2-signal-promoter'], check=True)
print("ENABLED")
subprocess.run(['systemctl', 'start', 'quantum-paper-trade-controller'], check=True)
print("STARTED: paper_trade_controller")
subprocess.run(['systemctl', 'start', 'quantum-layer2-signal-promoter'], check=True)
print("STARTED: layer2_signal_promoter")

# Final status
result = subprocess.run(['systemctl', 'is-active',
    'quantum-layer4-portfolio-optimizer',
    'quantum-apply-layer',
    'quantum-paper-trade-controller',
    'quantum-layer2-signal-promoter',
], capture_output=True, text=True)
print("STATUS:")
print(result.stdout)
print("=== DEPLOY COMPLETE ===")
