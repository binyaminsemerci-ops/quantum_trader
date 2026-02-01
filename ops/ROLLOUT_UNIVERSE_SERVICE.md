# Universe Service Rollout Steps

## Quick Deploy (VPS)

```bash
# 1. Pull latest code
cd /home/qt/quantum_trader
git pull

# 2. Copy config
sudo cp microservices/universe_service/universe-service.env.example /etc/quantum/universe-service.env
sudo chown qt:qt /etc/quantum/universe-service.env

# 3. Edit mode (testnet/mainnet) if needed
sudo nano /etc/quantum/universe-service.env

# 4. Install systemd unit
sudo cp ops/systemd/quantum-universe-service.service /etc/systemd/system/
sudo systemctl daemon-reload

# 5. Start service
sudo systemctl enable quantum-universe-service
sudo systemctl start quantum-universe-service

# 6. Verify
sudo systemctl status quantum-universe-service
bash ops/proof_universe.sh

# 7. Check logs
journalctl -u quantum-universe-service -f
```

## Verification Checklist

After deployment, verify:

- [ ] Service running: `systemctl is-active quantum-universe-service` returns `active`
- [ ] Redis key exists: `redis-cli EXISTS quantum:cfg:universe:active` returns `1`
- [ ] Symbol count reasonable: `redis-cli HGET quantum:cfg:universe:meta count` shows 400-800 for testnet
- [ ] Not stale: `redis-cli HGET quantum:cfg:universe:meta stale` returns `0`
- [ ] Proof script works: `bash ops/proof_universe.sh` shows 20 symbols
- [ ] Logs clean: `journalctl -u quantum-universe-service -n 50` shows no errors

## Integration with Existing Gates

Once Universe Service is running, update gate services to consume from Redis:

### Example: Position State Brain (P3.3)

Before (hardcoded):
```python
# /etc/quantum/position-state-brain.env
P33_ALLOWLIST=BTCUSDT,ETHUSDT,TRXUSDT,...
```

After (dynamic):
```python
# microservices/position_state_brain/main.py
import redis
import json

r = redis.Redis(decode_responses=True)
universe_json = r.get('quantum:cfg:universe:active')
universe = json.loads(universe_json)
self.symbols = set(universe['symbols'])  # Now dynamic!
```

**Note:** Gate integration is NOT done in this P0 task. This is documented for future reference.

## Rollback

If issues occur:

```bash
# Stop service
sudo systemctl stop quantum-universe-service
sudo systemctl disable quantum-universe-service

# Remove systemd unit
sudo rm /etc/systemd/system/quantum-universe-service.service
sudo systemctl daemon-reload

# Clean Redis keys (optional)
redis-cli DEL quantum:cfg:universe:active
redis-cli DEL quantum:cfg:universe:last_ok
redis-cli DEL quantum:cfg:universe:meta
```

Gates will continue using their hardcoded allowlists (no impact to trading).

## Monitoring

Add to monitoring:

- **Service health:** `systemctl is-active quantum-universe-service`
- **Staleness:** `redis-cli HGET quantum:cfg:universe:meta stale` should be `0`
- **Symbol count:** Track `redis-cli HGET quantum:cfg:universe:meta count` over time
- **Error state:** Alert if `redis-cli HGET quantum:cfg:universe:meta error` non-empty for > 5 minutes

## Performance Notes

- **Memory:** Service uses ~50MB RAM (Python + Redis client)
- **CPU:** Negligible (60s refresh interval, ~200ms per fetch)
- **Network:** ~100KB per fetch to Binance API
- **Redis:** 3 keys, ~500KB total (800 symbols × ~10 bytes + metadata)

## Testnet vs Mainnet

**Testnet:**
- Endpoint: `https://testnet.binancefuture.com/fapi/v1/exchangeInfo`
- Symbols: ~400-600 (subset of mainnet)
- Use for: Development, testing

**Mainnet:**
- Endpoint: `https://fapi.binance.com/fapi/v1/exchangeInfo`
- Symbols: ~800 (as of 2026)
- Use for: Production trading

Set `UNIVERSE_MODE=mainnet` in `/etc/quantum/universe-service.env` for production.
