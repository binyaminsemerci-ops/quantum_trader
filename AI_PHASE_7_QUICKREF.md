# ⚡ PHASE 7: TRADE JOURNAL & ANALYTICS - QUICK REFERENCE

**Fast commands for trade performance monitoring**

---

## 🚀 STARTUP COMMANDS

### Start Trade Journal
```bash
cd ~/quantum_trader
docker compose build trade-journal --no-cache
docker compose up -d trade-journal
```

### View Startup Logs
```bash
journalctl -u quantum_trade_journal.service --tail 30
```

### Check Health
```bash
systemctl list-units | grep trade_journal
```

---

## 📊 REPORT COMMANDS

### View Latest Report (Redis)
```bash
redis-cli GET latest_report | python3 -m json.tool
```

### View Latest Report (Dashboard API)
```bash
curl http://46.224.116.254:8501/report | python3 -m json.tool
```

### View Report File
```bash
docker exec quantum_trade_journal cat /app/reports/daily_report_$(date +%Y-%m-%d).json | python3 -m json.tool
```

### List All Reports
```bash
docker exec quantum_trade_journal ls -lah /app/reports/
```

### View Historical Reports
```bash
curl http://46.224.116.254:8501/reports/history | python3 -m json.tool
```

---

## 📈 METRICS COMMANDS

### Quick Performance Check
```bash
redis-cli GET latest_report | python3 -c "import sys,json; r=json.loads(sys.stdin.read()); print(f\"Trades: {r['total_trades']} | Win Rate: {r['win_rate_%']}% | Sharpe: {r['sharpe_ratio']} | Drawdown: {r['max_drawdown_%']}%\")"
```

### View Win Rate
```bash
redis-cli GET latest_report | python3 -c "import sys,json; print('Win Rate:', json.loads(sys.stdin.read())['win_rate_%'], '%')"
```

### View Sharpe Ratio
```bash
redis-cli GET latest_report | python3 -c "import sys,json; print('Sharpe Ratio:', json.loads(sys.stdin.read())['sharpe_ratio'])"
```

### View Max Drawdown
```bash
redis-cli GET latest_report | python3 -c "import sys,json; print('Max Drawdown:', json.loads(sys.stdin.read())['max_drawdown_%'], '%')"
```

### View Current Equity
```bash
redis-cli GET latest_report | python3 -c "import sys,json; r=json.loads(sys.stdin.read()); print(f\"Equity: ${r['current_equity']:,.2f} ({r['equity_change_%']:+.2f}%)\")"
```

---

## 🔍 MONITORING COMMANDS

### Live Logs (Follow Mode)
```bash
journalctl -u quantum_trade_journal.service -f
```

### Last Report Generation
```bash
journalctl -u quantum_trade_journal.service | grep "PERFORMANCE SUMMARY" -A 10 | tail -11
```

### Check Alert Status
```bash
redis-cli LRANGE journal_alerts 0 9 | python3 -m json.tool
```

### View Last Update Time
```bash
redis-cli GET journal_last_update
```

---

## 🔄 CONTROL COMMANDS

### Trigger Manual Report
```bash
docker exec quantum_trade_journal python journal_service.py
```

### Restart Journal Service
```bash
docker restart quantum_trade_journal
```

### Stop Journal Service
```bash
docker stop quantum_trade_journal
```

### Rebuild and Restart
```bash
cd ~/quantum_trader
docker compose build trade-journal --no-cache
docker restart quantum_trade_journal
```

---

## 📊 ANALYSIS COMMANDS

### Trade Count
```bash
redis-cli GET latest_report | python3 -c "import sys,json; r=json.loads(sys.stdin.read()); print(f\"Total: {r['total_trades']} | Wins: {r['winning_trades']} | Losses: {r['losing_trades']}\")"
```

### Average Trade PnL
```bash
redis-cli GET latest_report | python3 -c "import sys,json; print('Avg Trade:', json.loads(sys.stdin.read())['avg_trade_%'], '%')"
```

### Profit Factor
```bash
redis-cli GET latest_report | python3 -c "import sys,json; print('Profit Factor:', json.loads(sys.stdin.read())['profit_factor'])"
```

### Latest Trade Info
```bash
redis-cli GET latest_report | python3 -c "import sys,json; t=json.loads(sys.stdin.read())['latest_trade']; print(f\"{t['symbol']} {t['action']} {t['qty']} @ {t['price']} (PnL: {t['pnl']}%)\")"
```

---

## 📁 REPORT MANAGEMENT

### Backup Today's Report
```bash
docker exec quantum_trade_journal cat /app/reports/daily_report_$(date +%Y-%m-%d).json > ~/backups/report_$(date +%Y%m%d_%H%M%S).json
```

### Backup All Reports
```bash
docker exec quantum_trade_journal tar -czf /tmp/reports_backup.tar.gz /app/reports/
docker cp quantum_trade_journal:/tmp/reports_backup.tar.gz ~/backups/reports_$(date +%Y%m%d).tar.gz
```

### Clean Old Reports (keep last 30 days)
```bash
docker exec quantum_trade_journal find /app/reports/ -name "daily_report_*.json" -mtime +30 -delete
```

---

## 🚨 ALERT COMMANDS

### View All Alerts
```bash
redis-cli LRANGE journal_alerts 0 -1 | python3 -m json.tool
```

### Count Active Alerts
```bash
redis-cli LLEN journal_alerts
```

### Clear Alerts
```bash
redis-cli DEL journal_alerts
```

### Check for Critical Alerts
```bash
redis-cli LRANGE journal_alerts 0 -1 | grep -i "CRITICAL"
```

---

## 🔧 CONFIGURATION COMMANDS

### Change Update Interval (requires rebuild)
```bash
# Edit systemctl.yml
nano ~/quantum_trader/systemctl.yml
# Find REPORT_INTERVAL_HOURS and change value

# Rebuild
cd ~/quantum_trader
docker compose build trade-journal --no-cache
docker restart quantum_trade_journal
```

### Adjust Starting Equity
```bash
# Edit systemctl.yml
nano ~/quantum_trader/systemctl.yml
# Find STARTING_EQUITY and change value

# Restart
docker restart quantum_trade_journal
```

---

## 📊 ONE-LINER DASHBOARD

### Complete Performance Summary
```bash
echo "=== TRADE JOURNAL STATUS ===" && \
systemctl list-units --format "{{.Names}}: {{.Status}}" | grep trade_journal && \
echo -e "\n=== LATEST PERFORMANCE ===" && \
redis-cli GET latest_report | python3 -c "import sys,json; r=json.loads(sys.stdin.read()); print(f\"Trades: {r['total_trades']} | Win Rate: {r['win_rate_%']}% | Sharpe: {r['sharpe_ratio']} | Sortino: {r['sortino_ratio']} | Max DD: {r['max_drawdown_%']}% | PnL: {r['total_pnl_%']}%\")" && \
echo -e "\n=== RECENT ALERTS ===" && \
redis-cli LLEN journal_alerts | xargs -I {} echo "Active Alerts: {}"
```

---

## 🧪 TESTING COMMANDS

### Generate Test Trades
```bash
# Create winning trade
redis-cli RPUSH trade_log '{"symbol":"BTCUSDT","action":"BUY","qty":100,"price":50000,"confidence":0.75,"pnl":5.0,"timestamp":"2025-12-20T10:00:00","leverage":3,"paper":true,"testnet":true}'

# Create losing trade
redis-cli RPUSH trade_log '{"symbol":"ETHUSDT","action":"SELL","qty":50,"price":3500,"confidence":0.60,"pnl":-2.5,"timestamp":"2025-12-20T10:05:00","leverage":3,"paper":true,"testnet":true}'
```

### Verify Trade Count
```bash
redis-cli LLEN trade_log
```

### Trigger Report After Test
```bash
docker restart quantum_trade_journal && sleep 10 && journalctl -u quantum_trade_journal.service --tail 30
```

---

## 🔍 TROUBLESHOOTING QUICK FIXES

### Issue: No Report Generated
```bash
# Check if trade log has data
redis-cli LRANGE trade_log 0 5

# If empty, check auto-executor
systemctl list-units | grep auto_executor

# Check journal logs for errors
journalctl -u quantum_trade_journal.service --tail 50
```

### Issue: Metrics Look Wrong
```bash
# Verify trade log format
redis-cli LRANGE trade_log 0 2 | python3 -m json.tool

# Restart journal to recalculate
docker restart quantum_trade_journal
```

### Issue: Dashboard Not Showing Reports
```bash
# Test endpoint directly
curl http://localhost:8501/report

# Check if latest_report exists in Redis
redis-cli EXISTS latest_report

# Restart dashboard
docker restart quantum_governance_dashboard
```

---

## 📖 INTERPRETATION GUIDE

### Sharpe Ratio
```
> 2.0 = Excellent (institutional grade)
> 1.0 = Good (profitable with acceptable risk)
> 0.5 = Acceptable (needs improvement)
< 0.0 = Poor (losing money)
```

### Sortino Ratio
```
> 2.0 = Excellent (minimal downside risk)
> 1.5 = Good (well-managed downside)
> 1.0 = Acceptable
< 0.5 = High downside risk
```

### Win Rate
```
> 60% = Excellent
> 55% = Good
> 50% = Break-even threshold
< 50% = Losing strategy (unless high avg win)
```

### Max Drawdown
```
< 5%  = Excellent (very safe)
< 10% = Good (acceptable risk)
< 20% = Caution (high risk)
> 20% = Danger (review strategy immediately)
```

### Profit Factor
```
> 2.0 = Excellent (strong profitability)
> 1.5 = Good (sustainable profits)
> 1.0 = Break-even
< 1.0 = Losing money
```

---

## 📱 DASHBOARD URLS

### View Latest Report
```
http://46.224.116.254:8501/report
```

### View Report History
```
http://46.224.116.254:8501/reports/history
```

### Main Dashboard
```
http://46.224.116.254:8501
```

---

## 🛠️ COMMON WORKFLOWS

### Workflow 1: Daily Check
```bash
# Quick status
systemctl list-units | grep -E "(trade_journal|auto_executor)"

# View latest metrics
curl -s http://46.224.116.254:8501/report | python3 -c "import sys,json; r=json.load(sys.stdin); print(f\"Trades: {r['total_trades']} | Win: {r['win_rate_%']}% | Sharpe: {r['sharpe_ratio']} | DD: {r['max_drawdown_%']}%\")"

# Check for alerts
redis-cli LLEN journal_alerts
```

### Workflow 2: Weekly Review
```bash
# Get week summary
curl -s http://46.224.116.254:8501/report | python3 -m json.tool > weekly_report_$(date +%Y%m%d).json

# View all alerts from week
redis-cli LRANGE journal_alerts 0 -1 | python3 -m json.tool

# Backup reports
docker exec quantum_trade_journal tar -czf /tmp/weekly_backup.tar.gz /app/reports/
```

### Workflow 3: Performance Investigation
```bash
# Full report
journalctl -u quantum_trade_journal.service | grep "PERFORMANCE SUMMARY" -A 20 | tail -21

# Recent trades
redis-cli LRANGE trade_log 0 19 | python3 -m json.tool

# Check if any trades rejected
journalctl -u quantum_auto_executor.service | grep -i "rejected"
```

---

## 💡 PRO TIPS

### Tip 1: Monitor Sharpe Daily
Track Sharpe ratio daily. Drop below 1.0 means review strategy immediately.

### Tip 2: Watch Win Rate Trend
Win rate should be stable. Sudden drops indicate model drift or market regime change.

### Tip 3: Drawdown is King
Max drawdown is your biggest risk metric. Keep it under 10% at all costs.

### Tip 4: Report Frequency
- Testing: 1 hour intervals
- Production: 6 hour intervals
- High-frequency trading: Real-time metrics

### Tip 5: Backup Weekly
Always backup reports weekly. Critical for tax reporting and strategy analysis.

---

## 📚 RELATED COMMANDS

### View Auto Executor Status
```bash
journalctl -u quantum_auto_executor.service --tail 20
```

### Check Governance Dashboard
```bash
curl http://46.224.116.254:8501/status
```

### View Alert System
```bash
journalctl -u quantum_governance_alerts.service --tail 20
```

---

**Quick Reference Version:** 1.0  
**Last Updated:** 2025-12-20  
**For:** Phase 7 Trade Journal & Performance Analytics  

