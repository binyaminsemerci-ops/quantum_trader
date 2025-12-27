# 🚀 QUICK MONITOR - AI Trading Status

## Rask Statussjekk (PowerShell)

```powershell
# Full status
curl http://localhost:8000/health

# Kun AI status
curl http://localhost:8000/ai/live-status -H "X-Admin-Token: live-admin-token"
```

## Live Monitor (Auto-refresh)

Kjør dette for å se live oppdateringer:

```powershell
while($true) {
    Clear-Host
    Write-Host "=== AI TRADER - LIVE MONITOR ===" -ForegroundColor Cyan
    Write-Host "Tid: $(Get-Date -Format 'HH:mm:ss')`n"
    
    $h = curl http://localhost:8000/health 2>$null | ConvertFrom-Json
    $ai = curl http://localhost:8000/ai/live-status -H "X-Admin-Token: live-admin-token" 2>$null | ConvertFrom-Json
    
    Write-Host "Backend: $($h.status)" -ForegroundColor Green
    Write-Host "Scheduler: $(if($h.scheduler.running){'RUNNING'}else{'STOPPED'})" -ForegroundColor $(if($h.scheduler.running){'Green'}else{'Red'})
    
    Write-Host "`nAI Signaler:" -ForegroundColor Yellow
    Write-Host "  BUY:  $($ai.predictions.buy_signals)"
    Write-Host "  SELL: $($ai.predictions.sell_signals)"
    Write-Host "  HOLD: $($ai.predictions.hold_signals)"
    
    Write-Host "`nPosisjoner:" -ForegroundColor Yellow
    foreach($p in $h.risk.positions.positions) {
        Write-Host "  $($p.symbol): $($p.quantity) units (`$$([math]::Round($p.notional, 2)))"
    }
    Write-Host "  TOTAL: `$$([math]::Round($h.risk.positions.total_notional, 2))" -ForegroundColor Green
    
    Write-Host "`nNeste execution: $($h.scheduler.execution_job.next_run_time)" -ForegroundColor Cyan
    Write-Host "`nOppdateres om 30 sekunder..." -ForegroundColor Gray
    Start-Sleep -Seconds 30
}
```

## Er Alt OK?

✅ **Backend = healthy** → Alt OK!  
✅ **Scheduler = running** → Jobber kjører!  
✅ **AI signaler genereres** → AI er aktiv!  
✅ **Posisjoner vises** → Synkronisert med Binance!  

## Hva Er Normalt?

- **BUY/SELL = 0-2**: Dette er NORMALT! AI er konservativ.
- **HOLD = 8-10**: Dette er BRA! Unngår overtrading.
- **Posisjoner endres sakte**: Normalt - AI trader smart, ikke mye.

## Når AI Handler?

AI vil sende ordre når:
1. Market beveger seg nok (>0.1% momentum)
2. Prediction score > ±0.001
3. Confidence er høy nok
4. Risk limits tillater det

Dette skjer typisk 10-20% av tiden.

## Stoppe Systemet?

```powershell
# Stop backend
Get-Process | Where-Object {$_.Path -like "*python*" -and (Get-NetTCPConnection -OwningProcess $_.Id -ErrorAction SilentlyContinue | Where-Object LocalPort -eq 8000)} | Stop-Process -Force

# Emergency kill switch (stops trading but keeps backend running)
curl -X POST "http://localhost:8000/risk/kill-switch" -H "X-Admin-Token: live-admin-token"
```

## Start På Nytt?

```powershell
cd c:\quantum_trader\backend
Start-Process pwsh -ArgumentList "-NoProfile","-ExecutionPolicy","Bypass","-Command","cd c:\quantum_trader\backend; `$env:QT_AI_RETRAINING_ENABLED='1'; uvicorn main:app --host 0.0.0.0 --port 8000" -WindowStyle Minimized
```

---

**✨ Alt kjører automatisk - bare overvåk og nyt! 🚀**
