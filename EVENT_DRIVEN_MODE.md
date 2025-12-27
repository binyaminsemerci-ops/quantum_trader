# Event-Driven vs Scheduled Trading

Quantum Trader now supports **two trading modes**:

## 🎯 Event-Driven Mode (AI Signal-Based)
**Recommended for active trading**

AI continuously monitors the market and executes trades **when it detects strong opportunities**, regardless of time. No fixed schedules.

### How it works:
1. AI checks market every 30 seconds (configurable)
2. Generates signals with confidence scores (0.0 - 1.0)
3. Only trades when confidence >= threshold (default 0.65)
4. Enforces cooldown period between trades per symbol (default 5 min)
5. Respects risk limits and kill switch

### Configuration (via environment variables):
```bash
# Enable event-driven mode
QT_EVENT_DRIVEN_MODE=true

# Symbols to monitor (comma-separated)
QT_SYMBOLS=BTCUSDT,ETHUSDT,BNBUSDT,XRPUSDT,SOLUSDT

# Minimum confidence required to trade (0.0 - 1.0)
# Higher = fewer trades, but higher quality
QT_CONFIDENCE_THRESHOLD=0.65

# How often to check market (seconds)
QT_CHECK_INTERVAL=30

# Minimum time between trades for same symbol (seconds)
QT_COOLDOWN_SECONDS=300

# AI model sensitivity threshold
QT_XGB_THRESHOLD=0.0005
```

### When to use:
- ✅ You want AI to trade opportunistically based on market signals
- ✅ You want to catch moves 24/7 without waiting for scheduled intervals
- ✅ You trust AI confidence scores to filter noise
- ✅ Active markets with frequent opportunities

---

## ⏰ Scheduled Mode (Fixed Intervals)
**Traditional approach**

Executes trades at **fixed time intervals** (e.g., every 15 minutes), regardless of market conditions.

### How it works:
1. Scheduler runs execution cycle every N seconds (default 900 = 15 min)
2. AI generates signals for all symbols
3. Executes all non-HOLD signals
4. Waits for next scheduled cycle

### Configuration:
```bash
# Event-driven mode OFF (default)
QT_EVENT_DRIVEN_MODE=false

# Execution interval (seconds)
QUANTUM_TRADER_EXECUTION_SECONDS=900
```

### When to use:
- ✅ You want predictable, periodic rebalancing
- ✅ You want to limit trade frequency strictly
- ✅ Less active markets
- ✅ Backtesting and analysis with fixed intervals

---

## Comparison

| Feature | Event-Driven | Scheduled |
|---------|-------------|-----------|
| **Trading trigger** | AI confidence signals | Fixed time intervals |
| **Flexibility** | High - trades when opportunities arise | Low - only at scheduled times |
| **Trade frequency** | Variable (based on market) | Fixed (every N seconds) |
| **Missed opportunities** | Fewer - monitors continuously | More - only checks periodically |
| **Complexity** | Higher (confidence tuning) | Lower (simple interval) |
| **CPU usage** | Slightly higher (continuous checks) | Lower (periodic only) |

---

## Quick Start

### Enable Event-Driven Mode:
```powershell
# Windows PowerShell
$env:QT_EVENT_DRIVEN_MODE="true"
$env:QT_CONFIDENCE_THRESHOLD="0.65"
$env:QT_CHECK_INTERVAL="30"
./scripts/start-backend.ps1 -NewWindow
```

```bash
# Linux/Mac
export QT_EVENT_DRIVEN_MODE=true
export QT_CONFIDENCE_THRESHOLD=0.65
export QT_CHECK_INTERVAL=30
./scripts/deploy-vps.sh
```

### Check Mode:
```powershell
(Invoke-WebRequest -UseBasicParsing http://localhost:8000/health).Content | ConvertFrom-Json | Select-Object -ExpandProperty scheduler
```

Look for log line on startup:
- `🎯 Event-driven trading mode active: N symbols, confidence >= 0.XX`
- `⏰ Scheduled trading mode active (fixed intervals)`

---

## Tuning Tips

### Event-Driven Mode:

**Too many trades (over-trading):**
- ⬆️ Increase `QT_CONFIDENCE_THRESHOLD` (e.g., 0.70, 0.75)
- ⬆️ Increase `QT_COOLDOWN_SECONDS` (e.g., 600, 900)

**Too few trades (missing opportunities):**
- ⬇️ Decrease `QT_CONFIDENCE_THRESHOLD` (e.g., 0.60, 0.55)
- ⬇️ Decrease `QT_CHECK_INTERVAL` (e.g., 20, 15)
- ⬇️ Decrease `QT_XGB_THRESHOLD` (e.g., 0.0003, 0.0002)

**Monitoring:**
- Watch logs for `🎯 Strong [BUY/SELL] signal detected (confidence=X.XX)`
- Track trade frequency via `/health` endpoint
- Review confidence scores in trade logs

---

## Example Configurations

### Conservative (Quality over Quantity):
```bash
QT_EVENT_DRIVEN_MODE=true
QT_CONFIDENCE_THRESHOLD=0.75
QT_CHECK_INTERVAL=60
QT_COOLDOWN_SECONDS=600
QT_XGB_THRESHOLD=0.001
```

### Balanced (Default):
```bash
QT_EVENT_DRIVEN_MODE=true
QT_CONFIDENCE_THRESHOLD=0.65
QT_CHECK_INTERVAL=30
QT_COOLDOWN_SECONDS=300
QT_XGB_THRESHOLD=0.0005
```

### Aggressive (More Opportunities):
```bash
QT_EVENT_DRIVEN_MODE=true
QT_CONFIDENCE_THRESHOLD=0.55
QT_CHECK_INTERVAL=20
QT_COOLDOWN_SECONDS=180
QT_XGB_THRESHOLD=0.0003
```

---

## Migration from Scheduled to Event-Driven

1. **Backup current config:**
   ```bash
   cp .env .env.backup
   ```

2. **Add event-driven variables to `.env`:**
   ```bash
   echo "QT_EVENT_DRIVEN_MODE=true" >> .env
   echo "QT_CONFIDENCE_THRESHOLD=0.65" >> .env
   echo "QT_CHECK_INTERVAL=30" >> .env
   echo "QT_COOLDOWN_SECONDS=300" >> .env
   ```

3. **Restart backend:**
   ```powershell
   ./scripts/start-backend.ps1 -NewWindow
   ```

4. **Monitor for 1 hour:**
   - Check logs for signal detections
   - Verify trade frequency is acceptable
   - Adjust thresholds if needed

5. **Revert if issues:**
   ```bash
   cp .env.backup .env
   # Restart backend
   ```

---

## FAQ

**Q: Can I run both modes simultaneously?**
A: No, you must choose one mode. Event-driven disables the scheduler.

**Q: How do I know which mode is active?**
A: Check the startup log message or `/health` endpoint's `scheduler.config`.

**Q: Does event-driven mode respect risk limits?**
A: Yes! Both modes use the same RiskGuard and respect kill switch, max daily loss, etc.

**Q: What if AI generates weak signals all day?**
A: Event-driven mode won't trade if confidence < threshold. It stays idle until opportunities arise.

**Q: Can I adjust confidence threshold without restart?**
A: Not yet, but you can add a `/config/event-driven` endpoint to update live. Want me to add it?

---

Need help? Check logs with:
```powershell
docker compose logs -f backend-live  # VPS
# or
Get-Content logs/*.log -Tail 100  # Local
```
