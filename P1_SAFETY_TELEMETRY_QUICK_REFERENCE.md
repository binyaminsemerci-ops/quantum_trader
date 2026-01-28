# P1 Safety Telemetry - Quick Reference

## Service Management

```bash
# Status check
systemctl status quantum-safety-telemetry

# View logs (live)
journalctl -u quantum-safety-telemetry -f

# Restart
systemctl restart quantum-safety-telemetry
```

## Metrics Access

```bash
# Full metrics
curl http://127.0.0.1:9105/metrics

# Safety metrics only
curl -s http://127.0.0.1:9105/metrics | grep "^quantum_safety_"

# Trade metrics
curl -s http://127.0.0.1:9105/metrics | grep "^quantum_trade_"
```

## Key Metrics to Watch

| Metric | Description | Alert When |
|--------|-------------|------------|
| `quantum_safety_safe_mode` | 1 = SAFE MODE active | == 1 |
| `quantum_safety_faults_last_1h` | Fault count (last hour) | > 10 |
| `quantum_trade_intent_rate_per_min` | Publish rate | == 0 for 5+ min |
| `quantum_safety_redis_up` | Redis connectivity | == 0 |
| `quantum_safety_exporter_errors_total` | Exporter errors | increasing |

## Test Safe Mode Detection

```bash
# Enable safe mode (60s TTL)
redis-cli SET quantum:safety:safe_mode 1 EX 60

# Wait for sample cycle (15s)
sleep 18

# Check metric (should be 1.0)
curl -s http://127.0.0.1:9105/metrics | grep "quantum_safety_safe_mode "

# Clear safe mode
redis-cli DEL quantum:safety:safe_mode
```

## Files Reference

| File | Path |
|------|------|
| **Exporter** | `/home/qt/quantum_trader/microservices/safety_telemetry/main.py` |
| **Config** | `/etc/quantum/safety-telemetry.env` |
| **Service** | `/etc/systemd/system/quantum-safety-telemetry.service` |
| **Venv** | `/opt/quantum/venvs/safety-telemetry` |
| **Dashboard** | `/home/qt/quantum_trader/grafana/dashboards/quantum_safety_telemetry.json` |
| **Prometheus Config** | `/home/qt/quantum_trader/grafana/prometheus_scrape_config.yml` |

## Grafana Dashboard Import

1. Open Grafana UI
2. Dashboards → Import
3. Upload: `/home/qt/quantum_trader/grafana/dashboards/quantum_safety_telemetry.json`
4. Select Prometheus data source
5. Import

## Prometheus Integration

Add to `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'quantum_safety_telemetry'
    static_configs:
      - targets: ['localhost:9105']
    scrape_interval: 15s
```

Then reload: `systemctl reload prometheus`

## Troubleshooting

**Service not starting?**
```bash
# Check logs
journalctl -u quantum-safety-telemetry -n 50

# Test Python directly
/opt/quantum/venvs/safety-telemetry/bin/python3 \
  /home/qt/quantum_trader/microservices/safety_telemetry/main.py
```

**Metrics not updating?**
```bash
# Check Redis
redis-cli PING

# Check exporter errors
curl -s http://127.0.0.1:9105/metrics | grep exporter_errors_total
```

## Configuration Changes

Edit `/etc/quantum/safety-telemetry.env`:

```bash
# Change sample interval
SAMPLE_INTERVAL_SEC=10  # default: 15

# Change port
PORT=9106  # default: 9105

# Restart to apply
systemctl restart quantum-safety-telemetry
```

## Current Status

**Endpoint:** http://127.0.0.1:9105/metrics  
**Sample Interval:** 15 seconds  
**Memory Usage:** ~20 MB  
**Redis Load:** ~2 commands/sec  
**Auto-Restart:** Enabled  

**Deployment Date:** 2026-01-19 01:48 UTC  
**Status:** ✅ OPERATIONAL
