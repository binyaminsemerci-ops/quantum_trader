# Quantum Trader - Native Systemd Deployment

**NO DOCKER RUNTIME** - All services run natively via systemd.

## Architecture

```
quantum-trader.target (master)
├── quantum-core.target
│   └── redis-server.service (native)
├── quantum-ai.target
│   └── quantum-ai-engine.service
└── quantum-exec.target
    └── quantum-execution.service
```

## Installation

### 1. Install System Dependencies

```bash
apt-get update
apt-get install -y \
    redis-server \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    build-essential \
    uuid-runtime \
    bc
```

### 2. Create Virtual Environments

```bash
mkdir -p /opt/quantum/venvs
python3.12 -m venv /opt/quantum/venvs/ai-engine
python3.12 -m venv /opt/quantum/venvs/execution
```

### 3. Install Python Dependencies

```bash
# AI Engine
/opt/quantum/venvs/ai-engine/bin/pip install -r /home/qt/quantum_trader/ops/requirements-native-ai-engine.txt

# Execution Service
/opt/quantum/venvs/execution/bin/pip install -r /home/qt/quantum_trader/ops/requirements-native-execution.txt
```

### 4. Deploy Systemd Units

```bash
cp /home/qt/quantum_trader/ops/systemd/native/*.service /etc/systemd/system/
cp /home/qt/quantum_trader/ops/systemd/native/*.target /etc/systemd/system/
systemctl daemon-reload
```

### 5. Create Configuration

```bash
mkdir -p /etc/quantum
```

Create `/etc/quantum/ai-engine.env`:
```bash
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
TESTNET_MODE=true
LOG_LEVEL=INFO
```

Create `/etc/quantum/execution.env`:
```bash
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
TESTNET_MODE=true
LOG_LEVEL=INFO
# Governor keys (set via redis-cli):
# quantum:kill=1
# quantum:mode=TESTNET
# quantum:governor:execution=ENABLED
```

### 6. Initialize Governor

```bash
redis-cli SET quantum:kill 1
redis-cli SET quantum:mode TESTNET
redis-cli SET quantum:governor:execution ENABLED
```

### 7. Start Services

```bash
systemctl enable --now quantum-trader.target
```

## Governor Control

### Enable Trading (USE WITH CAUTION)

```bash
redis-cli SET quantum:kill 0
```

### Emergency Stop

```bash
redis-cli SET quantum:kill 1
```

### Check Status

```bash
redis-cli MGET quantum:kill quantum:mode quantum:governor:execution
```

## Manual Testing (Correct EventBus Envelope Format)

**IMPORTANT:** Do NOT use raw field XADD (e.g., `XADD stream * symbol BTCUSDT side BUY ...`).  
EventBus requires proper envelope structure.

### Correct Format

EventBus envelope requires:
- `event_type`: "trade.intent"
- `payload`: JSON string with signal data
- `correlation_id`: UUID or unique identifier
- `timestamp`: ISO 8601 format
- `source`: Identifier (e.g., "manual-test")
- `trace_id`: Empty or tracing ID

### Example: Manual Signal Injection

```bash
# Generate correlation_id and timestamp
CORRELATION_ID=$(uuidgen)
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%S.%6N+00:00")

# Build JSON payload (single line, escape quotes)
PAYLOAD='{"symbol":"OPUSDT","side":"BUY","position_size_usd":10.0,"leverage":1.0,"entry_price":0.3165,"stop_loss":0.3085875,"take_profit":0.32599500000000003,"confidence":0.72,"timestamp":"'$TIMESTAMP'","model":"manual","meta_strategy":"test","consensus_count":1,"total_models":1}'

# XADD with EventBus envelope
redis-cli XADD quantum:stream:events "*" \
  event_type "trade.intent" \
  payload "$PAYLOAD" \
  correlation_id "$CORRELATION_ID" \
  timestamp "$TIMESTAMP" \
  source "manual-test" \
  trace_id ""
```

### Test Scenarios

**Test 1: Verify BLOCKED (kill=1)**
```bash
redis-cli SET quantum:kill 1
# Inject signal (see above)
sleep 2
journalctl -u quantum-execution.service --since "10 seconds ago" | grep "BLOCKED"
# Expected: "[GOVERNOR] 🛑 BLOCKED T00000X - quantum:kill=1 (KILL MODE)"
```

**Test 2: Verify PASSED (kill=0)**
```bash
redis-cli SET quantum:kill 0
# Inject signal (see above)
sleep 3
journalctl -u quantum-execution.service --since "15 seconds ago" | grep "PASSED\|Executing"
# Expected: "[GOVERNOR] ✅ PASSED T00000X" and "[EXECUTION-V2] Executing: OPUSDT BUY ..."

# RESTORE SAFE STATE
redis-cli SET quantum:kill 1
```

### Automated E2E Test

Run the automated governor test:
```bash
bash /home/qt/quantum_trader/ops/tests/test_governor_e2e.sh
# Tests both BLOCKED and PASSED modes, restores safe state
```

## Monitoring

### Check Service Status

```bash
systemctl status quantum-trader.target
systemctl status quantum-ai-engine.service
systemctl status quantum-execution.service
```

### View Logs

```bash
# All services
journalctl -u quantum-trader.target -f

# AI Engine only
journalctl -u quantum-ai-engine.service -f

# Execution only
journalctl -u quantum-execution.service -f

# Governor activity
journalctl -u quantum-execution.service -f | grep GOVERNOR
```

### Health Checks

```bash
# AI Engine
curl http://localhost:8001/health

# Execution
curl http://localhost:8003/health
```

## Troubleshooting

### Service Won't Start

```bash
# Check logs
journalctl -u quantum-ai-engine.service -n 50 --no-pager

# Verify venv
ls -la /opt/quantum/venvs/ai-engine/bin/python3

# Test manually
/opt/quantum/venvs/ai-engine/bin/python3 -c "import fastapi, redis; print('OK')"
```

### Redis Connection Issues

```bash
# Check redis
systemctl status redis-server
redis-cli PING

# Check config
cat /etc/quantum/ai-engine.env
```

### Governor Not Blocking

```bash
# Verify keys
redis-cli MGET quantum:kill quantum:mode quantum:governor:execution

# Check execution logs
journalctl -u quantum-execution.service --since "1 minute ago" | grep GOVERNOR
```

## Maintenance

### Update Code

```bash
cd /home/qt/quantum_trader
git pull origin main

# Restart services
systemctl restart quantum-ai-engine.service
systemctl restart quantum-execution.service
```

### Update Dependencies

```bash
# AI Engine
/opt/quantum/venvs/ai-engine/bin/pip install -r /home/qt/quantum_trader/ops/requirements-native-ai-engine.txt --upgrade

# Execution
/opt/quantum/venvs/execution/bin/pip install -r /home/qt/quantum_trader/ops/requirements-native-execution.txt --upgrade

# Restart
systemctl restart quantum-trader.target
```

## Emergency Procedures

### Kill Switch (Immediate Stop)

```bash
redis-cli SET quantum:kill 1
# All new orders will be blocked immediately
```

### Full Stop

```bash
systemctl stop quantum-trader.target
```

### Full Restart

```bash
systemctl restart quantum-trader.target
```

### Rollback

```bash
cd /home/qt/quantum_trader
git checkout <previous-commit>
systemctl restart quantum-trader.target
```

## Security Hardening

Services include:
- `NoNewPrivileges=true` - Prevent privilege escalation
- `PrivateTmp=true` - Isolated /tmp
- `ProtectSystem=full` - Read-only /usr, /boot, /efi
- `ProtectHome=true` - Inaccessible /home
- `MemoryMax` - Resource limits (1-2GB per service)
- `Restart=always` - Auto-restart on failure
- `RestartSec=2` - 2-second delay between restarts

## Production Checklist

Before enabling trading (kill=0):
- [ ] Verify TESTNET mode: `redis-cli GET quantum:mode` → "TESTNET"
- [ ] Test BLOCKED: `redis-cli SET quantum:kill 1` → inject signal → verify blocked
- [ ] Test PASSED: `redis-cli SET quantum:kill 0` → inject signal → verify passed → restore kill=1
- [ ] Run E2E test: `bash /home/qt/quantum_trader/ops/tests/test_governor_e2e.sh`
- [ ] Verify paper trading: Check logs for "PAPER ORDER" entries
- [ ] Monitor for 24h in TESTNET with kill=1
- [ ] Review all execution logs for anomalies
- [ ] Confirm whitelist contains only approved symbols
