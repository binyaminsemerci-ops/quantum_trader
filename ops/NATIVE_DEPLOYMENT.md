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

## 📁 Directory Layout

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

## 📝 Notes

- **Governor Gate**: Implemented in service_v2.py (lines ~328-375)
- **Rate Limiting**: 6 signals/min with 120s symbol cooldown
- **TESTNET**: Max position size $10 USD
- **Native Deployment**: Zero Docker dependencies in runtime
- **Fail-Safe**: Governor check failure = BLOCK execution

---

**Created**: 2026-01-08  
**Deployment**: Hetzner VPS (46.224.116.254)  
**Mode**: TESTNET with Governor KILL protection
