# Universe Service - Quick Reference Card

## 🚀 Deploy (5 commands)

```bash
cd /home/qt/quantum_trader && git pull
sudo cp microservices/universe_service/universe-service.env.example /etc/quantum/universe-service.env
sudo cp ops/systemd/quantum-universe-service.service /etc/systemd/system/ && sudo systemctl daemon-reload
sudo systemctl enable --now quantum-universe-service
bash ops/proof_universe.sh
```

## 🔍 Verify

```bash
# Service status
sudo systemctl status quantum-universe-service

# Proof script (shows symbols, age, stale flag)
bash ops/proof_universe.sh

# Direct Redis check
redis-cli HGETALL quantum:cfg:universe:meta
```

## 📊 Monitor

```bash
# Live logs
journalctl -u quantum-universe-service -f

# Check staleness (should be 0)
redis-cli HGET quantum:cfg:universe:meta stale

# Check error (should be empty)
redis-cli HGET quantum:cfg:universe:meta error

# Symbol count
redis-cli HGET quantum:cfg:universe:meta count
```

## 🔧 Common Operations

```bash
# Restart service
sudo systemctl restart quantum-universe-service

# Force immediate refresh (restart triggers fetch)
sudo systemctl restart quantum-universe-service && sleep 2 && bash ops/proof_universe.sh

# View all symbols
redis-cli GET quantum:cfg:universe:active | jq -r '.symbols[]'

# Change mode (testnet ↔ mainnet)
sudo nano /etc/quantum/universe-service.env  # Edit UNIVERSE_MODE
sudo systemctl restart quantum-universe-service
```

## 🧪 Test Before Deploy

```bash
# Syntax check
python3 -m py_compile microservices/universe_service/main.py

# Mock output (shows expected structure)
python3 ops/test_universe_mock.py

# Deployment readiness
bash ops/check_universe_deployment_ready.sh
```

## 📖 Redis Keys

| Key | Type | Purpose |
|-----|------|---------|
| `quantum:cfg:universe:active` | string (JSON) | Current symbols (800 max) |
| `quantum:cfg:universe:last_ok` | string (JSON) | Last good fetch (fail-closed) |
| `quantum:cfg:universe:meta` | hash | Metadata (count, stale, error) |

## 🔐 Configuration

**File:** `/etc/quantum/universe-service.env`

```bash
UNIVERSE_MODE=testnet        # testnet | mainnet
UNIVERSE_REFRESH_SEC=60      # Fetch interval
UNIVERSE_MAX=800             # Safety cap
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_DB=0
HTTP_TIMEOUT_SEC=10
```

## ⚡ One-Liner Status

```bash
echo "Service: $(systemctl is-active quantum-universe-service) | Symbols: $(redis-cli HGET quantum:cfg:universe:meta count) | Stale: $(redis-cli HGET quantum:cfg:universe:meta stale) | Age: $(( $(date +%s) - $(redis-cli HGET quantum:cfg:universe:meta asof_epoch) ))s"
```

## 🧯 Rollback

```bash
sudo systemctl stop quantum-universe-service
sudo systemctl disable quantum-universe-service
sudo rm /etc/systemd/system/quantum-universe-service.service
sudo systemctl daemon-reload
```

## 🔗 Integration (Future P1)

```python
import redis, json
r = redis.Redis(decode_responses=True)
universe = json.loads(r.get('quantum:cfg:universe:active'))
symbols = universe['symbols']  # Replace hardcoded allowlists
```

## 📚 Documentation

- **Full guide:** [ops/ROLLOUT_UNIVERSE_SERVICE.md](ops/ROLLOUT_UNIVERSE_SERVICE.md)
- **README:** [ops/README.md](ops/README.md#universe-service-p0)
- **Summary:** [UNIVERSE_SERVICE_P0_COMPLETE.md](UNIVERSE_SERVICE_P0_COMPLETE.md)

## 🎯 Key Features

✅ **Fail-closed:** Preserves last_ok on errors  
✅ **Validated:** Regex + count caps + non-empty  
✅ **Bootstrap:** Recovers from last_ok on restart  
✅ **Minimal:** No HTTP server, Redis-only  
✅ **Safe:** Read-only, no trading logic  
✅ **Monitored:** Systemd journal + Redis meta  

## 📞 Troubleshooting

**Service won't start:**
```bash
journalctl -u quantum-universe-service -n 50
# Check: Python installed? Redis reachable? Config file exists?
```

**Stale flag = 1:**
```bash
redis-cli HGET quantum:cfg:universe:meta error
# Check: Binance API reachable? Correct endpoint for mode?
```

**No symbols:**
```bash
redis-cli GET quantum:cfg:universe:active | jq .
# Check: First fetch completed? Service running > 2s?
```
