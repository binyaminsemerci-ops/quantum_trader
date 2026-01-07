# Quantum Trader - Native Systemd Deployment

**Complete native deployment guide - no Docker runtime**

## ✅ Status

- **Docker**: Stopped and disabled
- **Redis**: Native redis-server.service (port 6379)
- **Python**: Virtual environments in /opt/quantum/venvs/
- **Services**: Native systemd services with Governor Gate protection

## 🎯 Architecture

```
quantum-trader.target (MASTER)
  ├── quantum-core.target
  │   └── redis-server.service (native)
  ├── quantum-ai.target
  │   └── quantum-ai-engine.service (venv: /opt/quantum/venvs/ai-engine)
  └── quantum-exec.target
      └── quantum-execution.service (venv: /opt/quantum/venvs/execution)
          └── [GOVERNOR GATE] Checks quantum:kill before EVERY order
```

## � Installation (One-Time Setup)

### Prerequisites
```bash
sudo apt update
sudo apt install -y redis-server python3.12 python3.12-venv python3-pip
```

### Create Virtual Environments
```bash
sudo mkdir -p /opt/quantum/venvs
sudo python3.12 -m venv /opt/quantum/venvs/ai-engine
sudo python3.12 -m venv /opt/quantum/venvs/execution
```

### Install Dependencies (Pinned Versions)
```bash
# AI Engine
sudo /opt/quantum/venvs/ai-engine/bin/pip install -r /home/qt/quantum_trader/ops/requirements-native-ai-engine.txt

# Execution Service
sudo /opt/quantum/venvs/execution/bin/pip install -r /home/qt/quantum_trader/ops/requirements-native-execution.txt
```

### Deploy Systemd Units
```bash
# Copy systemd files
sudo cp /home/qt/quantum_trader/ops/systemd/native/*.target /etc/systemd/system/
sudo cp /home/qt/quantum_trader/ops/systemd/native/*.service /etc/systemd/system/

# Create config directory
sudo mkdir -p /etc/quantum
sudo cp /home/qt/quantum_trader/ops/systemd/native/ai-engine.env /etc/quantum/
sudo cp /home/qt/quantum_trader/ops/systemd/native/execution.env /etc/quantum/

# Reload systemd
sudo systemctl daemon-reload
```

## �📁 Directory Layout

```
/home/qt/quantum_trader/          # Git repo
/opt/quantum/venvs/               # Python virtual environments
  ├── ai-engine/                  # AI Engine venv
  └── execution/                  # Execution venv
/etc/quantum/                     # Environment files
  ├── ai-engine.env               # AI Engine config
  └── execution.env               # Execution config (with governor refs)
/etc/systemd/system/              # Systemd units
  ├── quantum-trader.target       # Master target
  ├── quantum-core.target
  ├── quantum-ai.target
  ├── quantum-exec.target
  ├── quantum-ai-engine.service
  └── quantum-execution.service
```

## 🛡️ Governor Control

### Keys (Redis)
- `quantum:kill` - **1 = KILL** (block all orders), **0 = GO** (allow trading)
- `quantum:mode` - TESTNET | LIVE
- `quantum:governor:execution` - ENABLED | DISABLED

### Protected Services
- **execution** - Checks Governor Gate before ALL orders
- **ai-engine** - Rate-limited (MAX_SIGNALS_PER_MINUTE=6)

### Safe Defaults
```bash
# Initialize with KILL mode (safe)
redis-cli SET quantum:kill 1
redis-cli SET quantum:mode TESTNET
redis-cli SET quantum:governor:execution ENABLED
```

### Staged Bringup
```bash
# 1. Start core (Redis)
systemctl start quantum-core.target

# 2. Start AI layer
systemctl start quantum-ai.target

# 3. Start execution (with kill=1, will BLOCK orders)
systemctl start quantum-exec.target

# 4. Verify signals flow but execution is blocked
journalctl -u quantum-execution.service -f | grep -E "GOVERNOR|BLOCKED"

# 5. Enable trading (DANGEROUS - verify all systems first!)
redis-cli SET quantum:kill 0

# 6. Emergency stop
redis-cli SET quantum:kill 1
```

## 🚀 Deployment Commands

### Check Status
```bash
systemctl status quantum-trader.target
systemctl status quantum-ai-engine.service
systemctl status quantum-execution.service
```

### View Logs
```bash
journalctl -u quantum-ai-engine.service -f
journalctl -u quantum-execution.service -f
```

### Restart Services
```bash
systemctl restart quantum-ai.target
systemctl restart quantum-exec.target
```

### Governor Status
```bash
redis-cli MGET quantum:kill quantum:mode quantum:governor:execution
```

## ⚙️ Configuration

### AI Engine (/etc/quantum/ai-engine.env)
```bash
BINANCE_USE_TESTNET=true
REDIS_HOST=127.0.0.1
REDIS_PORT=6379

# Rate limiting (Jan 7 spam fix)
MIN_CONFIDENCE_THRESHOLD=0.65
MAX_SIGNALS_PER_MINUTE=6
SYMBOL_COOLDOWN_SECONDS=120

LOG_LEVEL=INFO
OTEL_ENABLED=true
```

### Execution (/etc/quantum/execution.env)
```bash
BINANCE_USE_TESTNET=true
REDIS_HOST=127.0.0.1
REDIS_PORT=6379

# Governor keys (checked in service_v2.py)
# quantum:kill - 1=KILL, 0=GO
# quantum:mode - TESTNET | LIVE
# quantum:governor:execution - ENABLED | DISABLED

# TESTNET Sizing
MAX_POSITION_SIZE_USD=10

LOG_LEVEL=INFO
OTEL_ENABLED=true
```

## 🔧 Maintenance

### Update Code
```bash
cd /home/qt/quantum_trader
git pull origin main

# Restart affected services
systemctl restart quantum-ai-engine.service
systemctl restart quantum-execution.service
```

### Update Dependencies
```bash
# AI Engine
sudo -u qt /opt/quantum/venvs/ai-engine/bin/pip install -U package_name

# Execution
sudo -u qt /opt/quantum/venvs/execution/bin/pip install -U package_name

# Restart after updates
systemctl restart quantum-ai-engine.service
```

### Check Governor Gate
```bash
# Verify Governor is checking kill switch
journalctl -u quantum-execution.service --since "1 minute ago" | grep GOVERNOR
# Expected: "[GOVERNOR] Checking... kill=1, mode=TESTNET, enabled=ENABLED"
#           "[GOVERNOR] 🛑 BLOCKED T000XXX - quantum:kill=1"
```

## 📊 Monitoring

### Service Health
```bash
# Overall system
systemctl status quantum-trader.target

# Individual services
systemctl status quantum-ai-engine.service --no-pager | head -20
systemctl status quantum-execution.service --no-pager | head -20
```

### Live Logs
```bash
# AI Engine signals
journalctl -u quantum-ai-engine.service -f | grep -E "Signal|confidence"

# Execution (with Governor)
journalctl -u quantum-execution.service -f | grep -E "GOVERNOR|BLOCKED|EXECUTE"
```

### Redis Streams
```bash
# Check stream lengths
redis-cli XLEN quantum:stream:trade.intent
redis-cli XLEN quantum:stream:execution.result

# Read latest signals
redis-cli XREVRANGE quantum:stream:trade.intent + - COUNT 5
```

## 🚨 Emergency Procedures

### Kill Switch (Immediate Stop)
```bash
redis-cli SET quantum:kill 1
# All new orders will be BLOCKED immediately
```

### Service Crash
```bash
# Check status
systemctl status quantum-execution.service

# View crash logs
journalctl -u quantum-execution.service -n 100 --no-pager

# Restart
systemctl restart quantum-execution.service
```

### Redis Issues
```bash
# Check Redis
systemctl status redis-server.service
redis-cli PING

# Restart Redis (will restart all dependent services)
systemctl restart redis-server.service
```

## 🧪 Manual Testing (Correct EventBus Envelope)

### ⚠️ DO NOT use raw XADD with symbol/side fields directly!

EventBus expects **envelope format** with these fields:
- `event_type` = "trade.intent"
- `payload` = JSON string (not raw fields!)
- `correlation_id` = UUID
- `timestamp` = ISO timestamp
- `source` = originating service
- `trace_id` = (optional)

### Stream Names
- **Stream**: `quantum:stream:{event_type}` → `quantum:stream:trade.intent`
- **Consumer Group**: `quantum:group:{service_name}:{event_type}` → `quantum:group:execution:trade.intent`

### Correct XADD Format

```bash
# Prepare JSON payload (escaped for redis-cli)
PAYLOAD='{"symbol":"OPUSDT","side":"BUY","position_size_usd":10.0,"leverage":1.0,"entry_price":0.3165,"stop_loss":0.3086,"take_profit":0.326,"confidence":0.72,"timestamp":"2026-01-08T00:00:00Z","model":"test","meta_strategy":"manual_test"}'

# Inject trade.intent event with proper envelope
redis-cli XADD quantum:stream:trade.intent "*" \
  event_type "trade.intent" \
  payload "$PAYLOAD" \
  correlation_id "test-$(uuidgen)" \
  timestamp "$(date -u +%Y-%m-%dT%H:%M:%S)" \
  source "ops-manual-test" \
  trace_id ""
```

### Test Scenarios

**Test 1: Governor BLOCKED (kill=1)**
```bash
# Set KILL mode
redis-cli SET quantum:kill 1

# Inject signal
redis-cli XADD quantum:stream:trade.intent "*" \
  event_type "trade.intent" \
  payload '{"symbol":"OPUSDT","side":"BUY","position_size_usd":10.0,"leverage":1.0,"entry_price":0.32,"stop_loss":0.31,"take_profit":0.33,"confidence":0.75,"timestamp":"2026-01-08T00:00:00Z"}' \
  correlation_id "test-blocked-001" \
  timestamp "$(date -u +%Y-%m-%dT%H:%M:%S)" \
  source "ops-test" \
  trace_id ""

# Verify BLOCKED in logs (within 3 seconds)
journalctl -u quantum-execution.service --since "10 seconds ago" | grep "🛑 BLOCKED"
```

**Test 2: Governor PASSED (kill=0)**
```bash
# Set GO mode (CAUTION!)
redis-cli SET quantum:kill 0

# Inject signal
redis-cli XADD quantum:stream:trade.intent "*" \
  event_type "trade.intent" \
  payload '{"symbol":"OPUSDT","side":"BUY","position_size_usd":10.0,"leverage":1.0,"entry_price":0.32,"stop_loss":0.31,"take_profit":0.33,"confidence":0.75,"timestamp":"2026-01-08T00:00:00Z"}' \
  correlation_id "test-passed-001" \
  timestamp "$(date -u +%Y-%m-%dT%H:%M:%S)" \
  source "ops-test" \
  trace_id ""

# Verify PASSED + execution in logs
journalctl -u quantum-execution.service --since "10 seconds ago" | grep -E "✅ PASSED|Executing"

# RESTORE SAFE STATE IMMEDIATELY
redis-cli SET quantum:kill 1
```

### Why Manual XADD May Fail

**❌ Wrong** (bypasses envelope):
```bash
redis-cli XADD quantum:stream:trade.intent "*" symbol BTCUSDT side BUY ...
# Result: event_data={} - no payload field!
```

**✅ Correct** (uses envelope):
```bash
redis-cli XADD quantum:stream:trade.intent "*" \
  event_type "trade.intent" \
  payload '{"symbol":"BTCUSDT","side":"BUY",...}' \
  correlation_id "..." timestamp "..." source "..."
```

## 📝 Notes

- **Governor Gate**: Implemented in service_v2.py (lines ~328-375)
- **Rate Limiting**: 6 signals/min with 120s symbol cooldown
- **TESTNET**: Max position size $10 USD
- **Native Deployment**: Zero Docker dependencies in runtime
- **Fail-Safe**: Governor check failure = BLOCK execution
- **EventBus**: Uses Redis Streams with envelope pattern (backend/core/event_bus.py)

---

**Created**: 2026-01-08  
**Deployment**: Hetzner VPS (46.224.116.254)  
**Mode**: TESTNET with Governor KILL protection
