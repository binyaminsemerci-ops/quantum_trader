# P3.9 Harvest Optimizer - Documentation

## Overview

The **Harvest Optimizer** is a READ-ONLY analytics microservice that consumes metrics from MetricPack Builder and Exit Intelligence to generate advanced performance analytics and regime-aware harvest recommendations.

**Version**: 1.0.0  
**Port**: 8052  
**Status**: Production Ready  

---

## Audit-Safe Guarantees

✅ **NO TRADING LOGIC CHANGES**
- Does not modify any trading services
- Does not write to apply.plan, trade.intent, or ai.decision streams
- Only reads from existing Prometheus metrics endpoints

✅ **READ-ONLY DATA ACCESS**
- Consumes metrics from MetricPack Builder (port 8051)
- Consumes metrics from Exit Intelligence (port 9109) if available
- Falls back gracefully if Exit Intelligence is not running

✅ **SAFE WRITES**
- Only writes to observability namespace (quantum:obs:*)
- Exports Prometheus metrics for monitoring
- Generates JSON/markdown reports

✅ **IDEMPOTENT & RESTARTABLE**
- Uses rolling windows for analysis
- No persistent state required
- Safe to restart at any time

---

## Architecture

```
┌─────────────────┐     ┌─────────────────┐
│  MetricPack     │────▶│                 │
│  Builder        │     │   Harvest       │
│  :8051/metrics  │     │   Optimizer     │
└─────────────────┘     │   :8052         │     ┌──────────────┐
                        │                 │────▶│ Prometheus   │
┌─────────────────┐     │  • Analytics    │     │ :9091        │
│  Exit           │────▶│  • Recommends   │     └──────────────┘
│  Intelligence   │     │  • Reports      │
│  :9109/metrics  │     │                 │     ┌──────────────┐
└─────────────────┘     └─────────────────┘────▶│ Grafana      │
                                                 │ Dashboard    │
                                                 └──────────────┘
```

---

## Metrics Calculated

### Core Performance Metrics

1. **Expectancy** (`quantum_ho_expectancy`)
   - Expected value per trade in USDT
   - Formula: `(avg_win × win_rate) - (avg_loss × loss_rate)`
   - Labels: `symbol`, `regime`, `exit_type`

2. **Profit Factor** (`quantum_ho_profit_factor`)
   - Ratio of gross profits to gross losses
   - Formula: `gross_profit / gross_loss`
   - Target: > 1.5 (good), > 2.0 (excellent)
   - Labels: `symbol`, `regime`

3. **Payoff Ratio** (`quantum_ho_payoff_ratio`)
   - Average win divided by average loss
   - Formula: `avg_win / avg_loss`
   - Labels: `symbol`, `regime`

4. **Exit Efficiency** (`quantum_ho_exit_efficiency`)
   - How much of MFE (Most Favorable Excursion) was captured
   - Formula: `exit_price / mfe_price`
   - Target: > 0.70 (trend), > 0.50 (chop)
   - Labels: `symbol`, `regime`

5. **Time Efficiency** (`quantum_ho_time_efficiency`)
   - Expectancy per hour (capital efficiency)
   - Formula: `expectancy / (avg_time_in_trade / 3600)`
   - Labels: `symbol`, `regime`

### Risk Metrics

6. **Adverse Recovery Rate** (`quantum_ho_adverse_recovery_rate`)
   - How much recovered from MAE (Most Adverse Excursion)
   - Formula: `(exit_pnl - mae) / (mfe - mae)`
   - Labels: `symbol`, `regime`

7. **Win Rate** (`quantum_ho_win_rate`)
   - Percentage of profitable trades
   - Labels: `symbol`, `regime`, `exit_type`

8. **Average Time in Trade** (`quantum_ho_avg_time_seconds`)
   - Mean duration of trades
   - Labels: `symbol`, `regime`

### Recommendation Metrics

9. **Recommendation Score** (`quantum_ho_recommendation_score`)
   - Confidence score for each recommendation (0.0 - 1.0)
   - Labels: `symbol`, `regime`, `rule`

10. **Regime Stability** (`quantum_ho_regime_stability`)
    - Persistence of current regime (0.0 - 1.0)
    - Labels: `symbol`, `regime`

---

## Recommendation Engine

The service generates **regime-aware recommendations** based on observed metrics:

### Trend Market Recommendations

**High Exit Efficiency (>0.75) + Positive Expectancy**
- **Action**: Widen partial ladder
- **Reason**: Trends persist, let winners run longer
- **Rule**: `partial_ladder`

**Long Time in Trade (>2h) + Low Time Efficiency**
- **Action**: Tighten time-stop
- **Reason**: Capital tied up too long
- **Rule**: `time_stop`

**Low Adverse Recovery (<0.5) + Large MAE**
- **Action**: Tighten stop-loss
- **Reason**: Not recovering from drawdown
- **Rule**: `stop_loss`

### Chop Market Recommendations

**Low Exit Efficiency (<0.5) + High MFE**
- **Action**: Accelerate partial_25
- **Reason**: Quick mean-reversion, take profits earlier
- **Rule**: `partial_25`

**High Payoff Ratio (>2.0) + Low Win Rate (<0.5)**
- **Action**: Tighten take-profit levels
- **Reason**: Targets too ambitious for chop
- **Rule**: `take_profit`

**Long Time in Trade (>1h) + Low Expectancy**
- **Action**: Tighten time-stop
- **Reason**: Chop requires faster exits
- **Rule**: `time_stop`

---

## API Endpoints

### GET /health
Health check endpoint.

**Response**:
```json
{
  "status": "ok",
  "service": "harvest_optimizer",
  "tracked_configs": 12,
  "last_update": 1706323200.0
}
```

### GET /metrics
Prometheus metrics in text format.

### GET /report
Generate comprehensive analytics report.

**Response**:
```json
{
  "generated_at": "2026-01-27T03:30:00",
  "analysis_window": "Last 200 trades",
  "performance_by_regime": {
    "BTCUSDT_trend": {
      "symbol": "BTCUSDT",
      "regime": "trend",
      "expectancy": 12.5,
      "profit_factor": 2.1,
      "win_rate": 0.62,
      "exit_efficiency": 0.78
    }
  },
  "recommendations": [
    {
      "symbol": "BTCUSDT",
      "regime": "trend",
      "rule_name": "partial_ladder",
      "action": "widen",
      "confidence": 0.85,
      "reason": "High exit efficiency (0.78) suggests trends persist. Widen partial ladder."
    }
  ],
  "top_performers": [...],
  "improvement_areas": [...],
  "summary": "..."
}
```

### GET /recommendations
Get current recommendations only.

**Response**:
```json
{
  "recommendations": [...],
  "count": 5,
  "timestamp": 1706323200.0
}
```

---

## Deployment

### 1. Install Dependencies

```bash
# On VPS
cd /home/qt/quantum_trader
source ~/quantum_trader_venv/bin/activate
pip install -r microservices/harvest_optimizer/requirements.txt
```

### 2. Deploy Configuration

```bash
sudo cp deploy/config/harvest-optimizer.env /etc/quantum/
```

**Edit `/etc/quantum/harvest-optimizer.env`** if needed:
- `SERVICE_PORT=8052` (default)
- `UPDATE_INTERVAL_SECONDS=60` (how often to update metrics)
- `ROLLING_WINDOW_SIZE=200` (trades to analyze)
- `SYMBOLS=BTCUSDT,ETHUSDT,TRXUSDT`

### 3. Install Systemd Service

```bash
sudo cp deploy/systemd/quantum-harvest-optimizer.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable quantum-harvest-optimizer
sudo systemctl start quantum-harvest-optimizer
```

### 4. Verify Service

```bash
# Check status
sudo systemctl status quantum-harvest-optimizer

# Check logs
sudo journalctl -u quantum-harvest-optimizer -f

# Test health endpoint
curl http://localhost:8052/health

# Test metrics endpoint
curl http://localhost:8052/metrics | head -40

# Get recommendations
curl http://localhost:8052/recommendations | jq .
```

### 5. Add to Prometheus

```bash
# Add to Prometheus config using Python yaml library
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 "python3 << 'PYEOF'
import yaml

with open('/etc/prometheus/prometheus.yml', 'r') as f:
    config = yaml.safe_load(f)

exists = any(job.get('job_name') == 'harvest_optimizer' for job in config['scrape_configs'])

if not exists:
    config['scrape_configs'].append({
        'job_name': 'harvest_optimizer',
        'static_configs': [{'targets': ['localhost:8052']}],
        'scrape_interval': '15s'
    })
    
    with open('/etc/prometheus/prometheus.yml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print('✅ harvest_optimizer added')
else:
    print('ℹ️  harvest_optimizer already exists')
PYEOF
"

# Validate and reload Prometheus
promtool check config /etc/prometheus/prometheus.yml
sudo systemctl reload prometheus

# Verify target
curl -s http://localhost:9091/api/v1/targets | jq '.data.activeTargets[] | select(.labels.job=="harvest_optimizer")'
```

### 6. Import Grafana Dashboard

1. Open Grafana: http://46.224.116.254:3000
2. Navigate to **Dashboards** → **Import**
3. Upload `deploy/grafana/harvest_optimizer_dashboard.json`
4. Select Prometheus data source
5. Click **Import**

---

## Dashboard Panels

The Grafana dashboard includes 12 panels:

1. **Expectancy by Regime** - Expected value per trade
2. **Profit Factor by Regime** - Gross profit/loss ratio
3. **Exit Efficiency** - How much of MFE captured
4. **Time Efficiency** - Expectancy per hour
5. **Win Rate by Regime** - Success rate
6. **Payoff Ratio** - Average win/loss
7. **Adverse Recovery Rate** - Recovery from MAE
8. **Average Time in Trade** - Trade duration
9. **Recommendation Scores** - Confidence levels
10. **Regime Stability** - Regime persistence gauge
11. **Reports Generated** - Total reports count
12. **Last Update** - Freshness indicator

---

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SERVICE_PORT` | 8052 | HTTP service port |
| `METRICPACK_URL` | http://localhost:8051/metrics | MetricPack source |
| `EXIT_INTEL_URL` | http://localhost:9109/metrics | Exit Intelligence source |
| `PROMETHEUS_URL` | http://localhost:9091 | Prometheus API |
| `UPDATE_INTERVAL_SECONDS` | 60 | Metric update frequency |
| `ROLLING_WINDOW_SIZE` | 200 | Trades in analysis window |
| `SMOOTHING_ALPHA` | 0.3 | Exponential smoothing factor |
| `SYMBOLS` | BTCUSDT,ETHUSDT,TRXUSDT | Tracked symbols |
| `LOG_LEVEL` | INFO | Logging verbosity |

---

## Interpreting Recommendations

### Confidence Levels

- **0.8 - 1.0**: High confidence (>40 trades), act on recommendation
- **0.5 - 0.8**: Moderate confidence (20-40 trades), monitor
- **0.0 - 0.5**: Low confidence (<20 trades), wait for more data

### Action Types

- **widen**: Increase spacing/delays in partial ladder
- **tighten**: Decrease spacing/delays
- **accelerate**: Trigger earlier
- **delay**: Trigger later
- **maintain**: Current settings optimal

### Using Recommendations

**DO**:
- Review recommendations in /report endpoint
- Cross-reference with dashboard metrics
- Test recommended changes in shadow mode first
- Document changes and measure impact

**DON'T**:
- Apply recommendations automatically without review
- Ignore confidence scores
- Change multiple rules simultaneously
- Override recommendations without understanding metrics

---

## Troubleshooting

### Service Not Starting

```bash
# Check logs
sudo journalctl -u quantum-harvest-optimizer -n 50

# Common issues:
# 1. Port already in use
sudo netstat -tlnp | grep 8052

# 2. MetricPack Builder not running
sudo systemctl status quantum-metricpack-builder

# 3. Python venv issues
source ~/quantum_trader_venv/bin/activate
pip install -r microservices/harvest_optimizer/requirements.txt
```

### No Metrics Showing

```bash
# Check if sources are accessible
curl http://localhost:8051/metrics | head -20
curl http://localhost:9109/metrics | head -20

# Check Prometheus target
curl -s http://localhost:9091/api/v1/targets | jq '.data.activeTargets[] | select(.labels.job=="harvest_optimizer")'

# Verify update loop is running
curl http://localhost:8052/health | jq .last_update
```

### Empty Recommendations

**Reason**: Insufficient trade data (< 10 trades per regime)

**Solution**: Wait for more trades to accumulate, or lower threshold in code.

---

## Safety Verification

### Verify Read-Only Behavior

```bash
# Check that service only reads from metrics endpoints
sudo strace -p $(pgrep -f harvest_optimizer) -e network 2>&1 | grep -E "(8051|9109|9091)"

# Should see GET requests only, no POST/PUT/DELETE

# Verify no writes to trading streams
redis-cli MONITOR | grep -E "(apply.plan|trade.intent|ai.decision)" &
# Let run for 1 minute - should see NO writes from harvest_optimizer
```

### Verify No Trading Impact

```bash
# Check that trading services are unmodified
git status | grep -E "(governor|permit|execution|brain)"
# Should show no changes

# Verify only observability namespace writes
redis-cli --scan --pattern "quantum:obs:*"
# Only harvest_optimizer keys should appear
```

---

## Performance

- **CPU**: < 5% during updates
- **Memory**: < 100 MB
- **Network**: < 1 KB/s (metrics polling)
- **Disk**: No persistent storage required

---

## Future Enhancements

1. **R-Multiple Distribution** - Requires risk proxy definition
2. **Fee/Slippage Estimator** - Requires execution data access
3. **Partial Sequence Scoring** - Conditional outcome tracking
4. **Regime Prediction** - ML-based regime forecasting
5. **A/B Test Framework** - Compare harvest strategies
6. **Automated Backtesting** - Historical recommendation validation

---

## Support

For issues or questions:
1. Check logs: `journalctl -u quantum-harvest-optimizer -f`
2. Verify dependencies: `systemctl status quantum-metricpack-builder quantum-exit-intelligence`
3. Review metrics: `curl http://localhost:8052/report | jq .summary`

---

**Last Updated**: 2026-01-27  
**Service**: P3.9 Harvest Optimizer  
**Status**: Production Ready ✅
