# IMPLEMENTATION SCRIPT - STAGED APPROACH (OPTION 2)

## PHASE 1: MODERATE SETTINGS (Hours 1-3)

### Changes to docker-compose.yml:

```yaml
# Line 29: Faster checks
- QT_CHECK_INTERVAL=7        # Was: 10 (30% faster)

# Line 30: Lower confidence threshold  
- QT_MIN_CONFIDENCE=0.30     # Was: 0.35 (+15% more signals)

# Line 35-37: Moderate position increase
- QT_MAX_NOTIONAL_PER_TRADE=300.0      # Was: 250 (+20%)
- QT_MAX_POSITION_PER_SYMBOL=300.0     # Was: 250
- QT_MAX_GROSS_EXPOSURE=3000.0         # Was: 2000

# Line 39: More concurrent positions
- QT_MAX_POSITIONS=10        # Was: 8 (+25%)

# Line 41: Keep TP same
- QT_TP_PCT=0.02            # Keep at 2.0% (realistic)

# Line 42: Keep SL same
- QT_SL_PCT=0.025           # Keep at 2.5%

# Line 43: Slightly wider trailing
- QT_TRAIL_PCT=0.012        # Was: 0.01 (+20% room)

# Line 44: Keep partial same
- QT_PARTIAL_TP=0.6         # Keep at 60%
```

### PowerShell Commands:

```powershell
# 1. BACKUP
Write-Host "📦 Backing up config..." -ForegroundColor Cyan
Copy-Item docker-compose.yml docker-compose.yml.backup
Write-Host "✅ Backup saved: docker-compose.yml.backup`n" -ForegroundColor Green

# 2. VERIFY CURRENT STATE
Write-Host "📊 Current settings:" -ForegroundColor Cyan
docker exec quantum_backend printenv | Select-String "QT_MAX_POSITIONS|QT_MAX_NOTIONAL|QT_CHECK_INTERVAL|QT_MIN_CONFIDENCE"

# 3. APPLY CHANGES (manual edit required)
Write-Host "`n⚠️  MANUAL STEP: Edit docker-compose.yml with Phase 1 values" -ForegroundColor Yellow
Write-Host "   Press ENTER when ready..." -ForegroundColor White
Read-Host

# 4. RESTART BACKEND
Write-Host "`n🔄 Restarting backend with new settings..." -ForegroundColor Cyan
docker-compose down
Start-Sleep 3
docker-compose up -d
Start-Sleep 5

# 5. VERIFY STARTUP
Write-Host "`n✅ Verifying backend startup..." -ForegroundColor Cyan
docker logs quantum_backend --tail 30

# 6. CHECK NEW SETTINGS
Write-Host "`n📊 New settings active:" -ForegroundColor Green
docker exec quantum_backend printenv | Select-String "QT_MAX_POSITIONS|QT_MAX_NOTIONAL|QT_CHECK_INTERVAL|QT_MIN_CONFIDENCE"

# 7. START MONITORING
Write-Host "`n🎯 Phase 1 ACTIVE - Monitoring for 3 hours..." -ForegroundColor Green
Write-Host "   Target by 06:00: `$30-40 P&L" -ForegroundColor Yellow
Write-Host "   Decision point: 06:00 (go Phase 2 if successful)`n" -ForegroundColor Cyan
```

---

## PHASE 2: AGGRESSIVE SETTINGS (Hours 4-6)

**ONLY if Phase 1 successful (P&L > $25 after 3 hours)**

### Additional Changes:

```yaml
# Line 29: Even faster
- QT_CHECK_INTERVAL=5        # Was: 7 (50% faster total)

# Line 30: Lower confidence
- QT_MIN_CONFIDENCE=0.25     # Was: 0.30 (+20% more signals)

# Line 35-37: Full aggressive
- QT_MAX_NOTIONAL_PER_TRADE=350.0      # Was: 300 (+17%)
- QT_MAX_POSITION_PER_SYMBOL=350.0     # Was: 300
- QT_MAX_GROSS_EXPOSURE=4200.0         # Was: 3000

# Line 39: Max positions
- QT_MAX_POSITIONS=12        # Was: 10 (+20%)

# Line 41: Higher TP
- QT_TP_PCT=0.025           # Was: 0.02 (+25% per win)
```

### PowerShell Commands:

```powershell
# 1. CHECK PHASE 1 RESULTS
Write-Host "📊 Phase 1 Results (after 3 hours):" -ForegroundColor Cyan
$health = curl -s http://localhost:8000/health | ConvertFrom-Json
Write-Host "   P&L: `$$($health.total_pnl)" -ForegroundColor $(if ($health.total_pnl -gt 25) { "Green" } else { "Red" })
Write-Host "   Trades: $($health.total_trades)" -ForegroundColor White

# 2. DECISION POINT
if ($health.total_pnl -gt 25) {
    Write-Host "`n✅ Phase 1 SUCCESS - Proceeding to Phase 2!" -ForegroundColor Green
    Write-Host "   Applying aggressive settings...`n" -ForegroundColor Yellow
    
    # 3. APPLY PHASE 2 CHANGES (manual edit)
    Write-Host "⚠️  MANUAL STEP: Edit docker-compose.yml with Phase 2 values" -ForegroundColor Yellow
    Write-Host "   Press ENTER when ready..." -ForegroundColor White
    Read-Host
    
    # 4. RESTART
    docker-compose down
    Start-Sleep 3
    docker-compose up -d
    Start-Sleep 5
    
    Write-Host "`n🎯 Phase 2 ACTIVE - Target: Additional `$40-60" -ForegroundColor Green
} else {
    Write-Host "`n⚠️  Phase 1 BELOW TARGET - STAYING MODERATE" -ForegroundColor Yellow
    Write-Host "   Continuing with Phase 1 settings..." -ForegroundColor White
}
```

---

## PHASE 3: SCALING (Hours 7-9)

**Adjust based on cumulative results**

### Decision Tree:

```
IF P&L > $80:
  → SCALE BACK to conservative
  → Protect gains
  → Target: Maintain +$80-100

ELSE IF P&L $50-80:
  → MAINTAIN Phase 2 settings
  → Push for $100 target
  
ELSE IF P&L $20-50:
  → MAINTAIN or INCREASE risk slightly
  → Manual intervention possible
  
ELSE IF P&L < $20:
  → EMERGENCY REVERT
  → Stop aggressive trading
  → Manual review required
```

### PowerShell Commands:

```powershell
# 1. CHECK PHASE 2 RESULTS (after 6 hours)
Write-Host "📊 Phase 2 Results (after 6 hours total):" -ForegroundColor Cyan
$health = curl -s http://localhost:8000/health | ConvertFrom-Json
Write-Host "   P&L: `$$($health.total_pnl)" -ForegroundColor $(if ($health.total_pnl -gt 70) { "Green" } elseif ($health.total_pnl -gt 50) { "Yellow" } else { "Red" })

# 2. DECISION
if ($health.total_pnl -gt 80) {
    Write-Host "`n🎉 AHEAD OF TARGET - Scaling back to protect gains" -ForegroundColor Green
    # Revert to Phase 1 moderate settings
    
} elseif ($health.total_pnl -gt 50) {
    Write-Host "`n⚠️  ON TRACK - Maintaining aggressive settings" -ForegroundColor Yellow
    # Keep Phase 2
    
} else {
    Write-Host "`n🚨 BEHIND TARGET - Manual review required" -ForegroundColor Red
    # Consider emergency stop or manual boost
}
```

---

## MONITORING COMMANDS

### Real-time P&L Check (run every 30 min):

```powershell
$health = curl -s http://localhost:8000/health | ConvertFrom-Json
Write-Host "`n💰 Current Status:" -ForegroundColor Cyan
Write-Host "   Time: $(Get-Date -Format 'HH:mm')" -ForegroundColor White
Write-Host "   P&L: `$$($health.total_pnl)" -ForegroundColor $(if ($health.total_pnl -gt 0) { "Green" } else { "Red" })
Write-Host "   Positions: $($health.open_positions)/12" -ForegroundColor White
Write-Host "   Total Trades: $($health.total_trades)`n" -ForegroundColor White
```

### Recent Trades:

```powershell
docker logs quantum_backend --tail 100 | Select-String "filled|CLOSED|TP triggered|SL triggered" | Select-Object -Last 20
```

### Live Streaming:

```powershell
docker logs quantum_backend -f | Select-String "Creating order|filled|P&L"
```

---

## EMERGENCY STOP

If P&L drops below -$50:

```powershell
Write-Host "🚨 EMERGENCY STOP ACTIVATED" -ForegroundColor Red

# 1. Stop backend
docker stop quantum_backend

# 2. Close all positions (via Binance API)
python close_all_positions.py

# 3. Revert config
Copy-Item docker-compose.yml.backup docker-compose.yml

# 4. Restart conservative
docker-compose up -d

Write-Host "✅ Reverted to conservative settings" -ForegroundColor Green
```

---

## SUCCESS CHECKPOINTS

**04:00 (1 hour):**
- Expected: 2-3 trades, $15-25 P&L
- Action: If 0 trades → check logs for issues

**06:00 (3 hours - END OF PHASE 1):**
- Expected: 6-8 trades, $30-40 P&L
- Action: Decide Phase 2 or stay moderate

**09:00 (6 hours - END OF PHASE 2):**
- Expected: 12-15 trades, $70-100 P&L
- Action: Scale back if ahead, maintain if on track

**12:00 (9 hours - FINAL):**
- Target: 18-24 trades, $100+ P&L ✅
- Success: P&L > $80 considered success

---

**Ready to start Phase 1?**
