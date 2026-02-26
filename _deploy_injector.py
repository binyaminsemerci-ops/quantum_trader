#!/usr/bin/env python3
"""Deploy signal injector to VPS as a systemd service"""
import subprocess, sys

SCRIPT_DST = "/opt/quantum/signal_injector.py"
SERVICE = "quantum-signal-injector"

SERVICE_UNIT = f"""[Unit]
Description=QuantumTrader Signal Injector - Redis stream publisher
After=network.target redis.service quantum-ai-engine.service
Wants=redis.service

[Service]
Type=simple
User=root
WorkingDirectory=/opt/quantum
EnvironmentFile=/etc/quantum/autonomous-trader.env
Environment=QT_SIGNAL_SYMBOLS=ETHUSDT,BTCUSDT,SOLUSDT,XRPUSDT,BNBUSDT,ADAUSDT,SUIUSDT,LINKUSDT,AVAXUSDT,LTCUSDT,DOTUSDT,NEARUSDT
Environment=QT_INJECT_INTERVAL=60
Environment=QT_INJECT_MIN_MOVE=0.003
Environment=QT_INJECT_CONFIDENCE=0.68
Environment=REDIS_URL=redis://127.0.0.1:6379
ExecStart=/opt/quantum/venvs/ai-engine/bin/python {SCRIPT_DST}
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""

SERVICE_PATH = f"/etc/systemd/system/{SERVICE}.service"

cmds = [
    # Copy script
    f"cp /tmp/signal_injector.py {SCRIPT_DST}",
    f"chmod +x {SCRIPT_DST}",
    # Write service file
    f"cat > {SERVICE_PATH} << 'EOFSERVICE'\n{SERVICE_UNIT}\nEOFSERVICE",
    # Reload + enable + start
    "systemctl daemon-reload",
    f"systemctl enable {SERVICE}",
    f"systemctl restart {SERVICE}",
    # Wait 5s then check
    "sleep 5",
    f"systemctl status {SERVICE} --no-pager -l | head -30",
    # Check Redis right after
    "sleep 5",
    "python3 -c \""
    "import redis, json; r=redis.Redis(host='127.0.0.1',port=6379,decode_responses=True); "
    "msgs=r.xrevrange('quantum:stream:ai.signal_generated', count=5); "
    "[print(m[0], json.loads(m[1].get('payload','{}'))['symbol'], "
    "json.loads(m[1].get('payload','{}'))['action'], m[1].get('timestamp','')) for m in msgs]"
    "\""
]

full_cmd = " && ".join(cmds)
print(f"[DEPLOY] Running: {len(cmds)} steps")
print(full_cmd[:200], "...")

# Actually run via SSH from caller
print("DEPLOY_CMD_READY")
print(full_cmd)
