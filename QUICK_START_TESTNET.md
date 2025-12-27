# ⚡ QUICK START - TESTNET TRADING

## 🚀 START SYSTEM (5 kommandoer)

```powershell
# 1. STOPP EKSISTERENDE
Get-Process python* -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

# 2. SETT MILJØVARIABLE
$env:PYTHONPATH='C:\quantum_trader'
$env:QT_EVENT_DRIVEN_MODE='true'
$env:QT_SYMBOLS='BTCUSDT,SOLUSDT'
$env:USE_BINANCE_TESTNET='true'
$env:QT_POSITION_MONITOR='false'

# 3. START BACKEND
cd C:\quantum_trader
python -m uvicorn backend.main:app --port 8000 --host 0.0.0.0
```

## ✅ VERIFISER

```powershell
# Health check
Invoke-RestMethod -Uri "http://localhost:8000/health"

# Live logs
Get-Content C:\quantum_trader\logs\quantum_trader.log -Wait | Select-String "FULL LIVE|Strong signals|REJECTED"
```

## 🎯 KONFIGURASJON

| Parameter | Fil | Linje | Verdi | Beskrivelse |
|-----------|-----|-------|-------|-------------|
| **Orchestrator threshold** | `backend/services/orchestrator_config.py` | 28 | `0.30` | SAFE profile base confidence |
| **TradeFilter threshold** | `backend/config/risk_management.py` | 202 | `0.40` | Min confidence for trades |
| **Trend alignment** | `backend/config/risk_management.py` | ~210 | `True` | Block trades against trend |

## 🔧 JUSTER FOR FLERE TRADES

### Option 1: Reduser thresholds
```python
# orchestrator_config.py, line 28
"base_confidence": 0.25,  # Fra 0.30

# risk_management.py, line 202
min_confidence: float = field(default=0.35)  # Fra 0.40
```

### Option 2: Deaktiver trend alignment (IKKE ANBEFALT)
```python
# risk_management.py, line ~210
require_trend_alignment: bool = field(default=False)
```

## 📊 FORVENTET OUTPUT

```
✅ FULL LIVE MODE - Policy ENFORCED
📋 Policy Controls: allow_trades=True, min_conf=0.42, position_limits=ACTIVE

Signal: SOLUSDT SELL (conf=0.47)
✅ PASSED Orchestrator (0.47 > 0.42)
❌ REJECTED TradeFilter: SHORT against trend
```

## 🐛 FEILSØKING

| Problem | Løsning |
|---------|---------|
| Port 8000 occupied | `Get-NetTCPConnection -LocalPort 8000 \| % { Stop-Process -Id $_.OwningProcess -Force }` |
| Ingen trades | Sjekk logs for rejection reason (trend/confidence) |
| System crashes | Verifiser at `QT_POSITION_MONITOR='false'` |
| API errors | Sjekk at `USE_BINANCE_TESTNET='true'` |

## 📝 SE FULL DOKUMENTASJON
`TESTNET_TRADING_FIXES_LOG.md` - Komplett fikslogg med alle detaljer
