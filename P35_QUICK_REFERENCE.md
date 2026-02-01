# P3.5 Quick Reference Card

## Deploy in 2 Minutes

```bash
# SSH to VPS
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254

# Deploy
cd /home/qt/quantum_trader && bash deploy_p35.sh
```

---

## Check Status

```bash
# Service running?
systemctl status quantum-p35-decision-intelligence

# Any lag?
redis-cli XPENDING quantum:stream:apply.result p35_decision_intel

# Health status
redis-cli HGETALL quantum:p35:status
```

---

## View Analytics

```bash
# Top skip reasons (last 5 min)
redis-cli ZREVRANGE quantum:p35:reason:top:5m 0 20 WITHSCORES

# Decision distribution (last 5 min)
redis-cli HGETALL quantum:p35:decision:counts:5m

# All buckets
redis-cli KEYS "quantum:p35:bucket:*"
```

---

## Common Queries

```bash
# High skip rate reason
redis-cli ZREVRANGE quantum:p35:reason:top:5m 0 1 WITHSCORES

# Blocked by risk filter
redis-cli ZREVRANGE quantum:p35:reason:top:5m 0 100 | grep kill_score

# Not in allowlist
redis-cli ZREVRANGE quantum:p35:reason:top:5m 0 100 | grep not_in_allowlist

# Service processing?
redis-cli HGET quantum:p35:status processed_total  # Should increase
```

---

## Troubleshooting

| Issue | Check | Fix |
|-------|-------|-----|
| Service down | `systemctl status ...` | Restart: `systemctl restart ...` |
| High pending | `XPENDING quantum:stream:apply.result p35_decision_intel` | `systemctl restart quantum-p35-decision-intelligence` |
| No analytics | Wait 1+ min for data | Check: `redis-cli KEYS "quantum:p35:*"` |
| Slow processing | `journalctl -u quantum-p35... \| head -20` | Check CPU/memory usage |

---

## Files

| File | Purpose |
|------|---------|
| `microservices/decision_intelligence/main.py` | Service code |
| `/etc/quantum/p35-decision-intelligence.env` | Configuration |
| `/etc/systemd/system/quantum-p35-decision-intelligence.service` | Systemd unit |
| `scripts/proof_p35_decision_intelligence.sh` | Verify deployment |
| `deploy_p35.sh` | One-command deploy |

---

## Redis Keys (Output)

```
quantum:p35:bucket:YYYYMMDDHHMM        Per-minute data (TTL: 48h)
quantum:p35:decision:counts:1m/5m/15m/1h   Decision distribution (TTL: 24h)
quantum:p35:reason:top:1m/5m/15m/1h        Top reasons ZSET (TTL: 24h)
quantum:p35:status                         Service health HASH (persistent)
```

---

## Service Logs

```bash
# Follow live
journalctl -u quantum-p35-decision-intelligence -f

# Last 50 lines
journalctl -u quantum-p35-decision-intelligence -n 50

# Since service start
journalctl -u quantum-p35-decision-intelligence --since today
```

---

## Configuration

File: `/etc/quantum/p35-decision-intelligence.env`

```
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
LOG_LEVEL=INFO
ENABLE_SYMBOL_BREAKDOWN=true
```

---

## Windows Available

- **1m**: 1-minute rolling window
- **5m**: 5-minute rolling window
- **15m**: 15-minute rolling window
- **1h**: 1-hour rolling window

Each computed every ~60 seconds.

---

## Success Indicators

✅ Service running: `systemctl is-active quantum-p35-decision-intelligence`  
✅ Processing messages: `HGET quantum:p35:status processed_total` increases  
✅ ACKing working: `XPENDING quantum:stream:apply.result p35_decision_intel` ≈ 0  
✅ Analytics available: `KEYS quantum:p35:decision:counts:5m` returns data  

---

## One-Liner Checks

```bash
# All systems go?
systemctl is-active quantum-p35-decision-intelligence && redis-cli PING && echo "✅ ALL GOOD"

# Processing rate
redis-cli HGET quantum:p35:status processed_total

# No backlog?
[ "$(redis-cli XPENDING quantum:stream:apply.result p35_decision_intel | head -1)" = "0" ] && echo "✅ CLEAR" || echo "⚠️ BACKLOG"

# Top reason why skipping
redis-cli ZREVRANGE quantum:p35:reason:top:5m 0 0

# Decisions this window
redis-cli HGETALL quantum:p35:decision:counts:5m | tr '\n' ' ' | sed 's/ /\n/2g'
```

---

## Performance (Expected)

- **Throughput**: 1,000+ decisions/sec
- **CPU**: ~5-10% (limit: 20%)
- **Memory**: ~50-100MB (limit: 256MB)
- **Latency**: <1ms buckets, ~500ms snapshots
- **Storage**: ~50MB per 24 hours

---

**Quick Link**: [Comprehensive Guide](./AI_P35_DEPLOYMENT_GUIDE.md)
