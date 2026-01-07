# Quantum Trader Service Matrix

Auto-generated from code analysis.

## Service Overview

| Service | Port | Target | Health | Streams Read | Streams Write |
|---------|------|--------|--------|--------------|---------------|
| ai_engine | 8001 | ai | ✅ | - | - |
| clm | 8008 | ai | ❌ | - | - |
| eventbus_bridge | 8009 | obs | ❌ | - | - |
| execution | 8003 | exec | ✅ | - | - |
| exposure_balancer | 8010 | exec | ❌ | - | quantum:stream:executor.commands, quantum:stream:exposure.alerts |
| portfolio_intelligence | 8006 | state | ✅ | - | - |
| position_monitor | 8005 | state | ❌ | - | - |
| risk_safety | 8007 | brains | ✅ | - | - |
| rl_training | 8002 | ai | ✅ | - | - |
| trading_bot | 8011 | unknown | ✅ | - | - |

## Target Hierarchy

```
quantum-trader.target (MASTER)
  ├── quantum-core.target (redis, market_publisher)
  ├── quantum-state.target (positions, portfolio, pnl) [Requires: core]
  ├── quantum-brains.target (ceo, risk, strategy) [Requires: state]
  ├── quantum-ai.target (ai-engine, rl, clm) [Requires: state]
  ├── quantum-exec.target (execution, balancer) [Requires: brains, ai]
  └── quantum-obs.target (eventbus) [Requires: core]
```

## Governor Control

**Keys:**
- `quantum:kill` - 1=KILL (block all), 0=GO (allow trading)
- `quantum:mode` - TESTNET | LIVE
- `quantum:governor:execution` - ENABLED | DISABLED

**Protected Services:**
- `execution` - Checks governor before ALL orders
- `ai_engine` - Rate-limited signal generation
- `rl_training` - TESTNET sizing cap ($10 max)

## Deployment Commands

```bash
# Initialize governor (safe defaults)
./scripts/ops/governor_init.sh

# Staged bringup
systemctl start quantum-core.target
systemctl start quantum-state.target quantum-ai.target
systemctl start quantum-exec.target  # Will block if kill=1

# Verify blocking
redis-cli GET quantum:kill  # Should be 1
docker logs -f quantum_execution  # Should see "🛑 BLOCKED"

# Enable trading (DANGEROUS - verify all systems first)
redis-cli SET quantum:kill 0

# Emergency stop
redis-cli SET quantum:kill 1
```

## Service Details

### ai_engine

**Port:** 8001  
**Entrypoint:** main.py  
**Health Check:** Yes  
**Redis Keys:**  
- `quantum:consensus:signal`  
- `quantum:evolution:mutated`  
- `quantum:evolution:rankings`  
- `quantum:evolution:retrain_count`  
- `quantum:evolution:selected`  

### clm

**Port:** 8008  
**Entrypoint:** main.py  
**Health Check:** No  

### eventbus_bridge

**Port:** 8009  
**Entrypoint:** main.py  
**Health Check:** No  

### execution

**Port:** 8003  
**Entrypoint:** main.py  
**Health Check:** Yes  

### exposure_balancer

**Port:** 8010  
**Entrypoint:** service.py  
**Health Check:** No  
**Writes Streams:**  
- `quantum:stream:executor.commands`  
- `quantum:stream:exposure.alerts`  
**Redis Keys:**  
- `quantum:cross:divergence`  
- `quantum:margin:total`  
- `quantum:meta:confidence`  

### portfolio_intelligence

**Port:** 8006  
**Entrypoint:** main.py  
**Health Check:** Yes  

### position_monitor

**Port:** 8005  
**Entrypoint:** main.py  
**Health Check:** No  

### risk_safety

**Port:** 8007  
**Entrypoint:** main.py  
**Health Check:** Yes  

### rl_training

**Port:** 8002  
**Entrypoint:** main.py  
**Health Check:** Yes  

### trading_bot

**Port:** 8011  
**Entrypoint:** main.py  
**Health Check:** Yes  

