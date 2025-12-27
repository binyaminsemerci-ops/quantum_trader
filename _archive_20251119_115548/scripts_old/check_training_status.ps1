#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Sjekker status på continuous training systemet
#>

Write-Host ""
Write-Host "📊 CONTINUOUS TRAINING STATUS" -ForegroundColor Cyan
Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Sjekk backend status
Write-Host "🔍 Sjekker backend..." -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 5 -ErrorAction Stop
    Write-Host "✅ Backend kjører" -ForegroundColor Green
} catch {
    Write-Host "❌ Backend ikke tilgjengelig" -ForegroundColor Red
    exit 1
}

# Sjekk database samples
Write-Host ""
Write-Host "🔍 Sjekker training samples..." -ForegroundColor Yellow
$sampleCheck = docker exec quantum_backend python -c @"
from backend.database import SessionLocal
from backend.models.ai_training import AITrainingSample
db = SessionLocal()
total = db.query(AITrainingSample).count()
with_outcome = db.query(AITrainingSample).filter(AITrainingSample.outcome_known == True).count()
print(f'{total},{with_outcome}')
db.close()
"@ 2>$null

if ($sampleCheck -match "(\d+),(\d+)") {
    $total = $Matches[1]
    $ready = $Matches[2]
    Write-Host "   Total samples: $total" -ForegroundColor White
    Write-Host "   Ready for training: $ready" -ForegroundColor $(if ($ready -gt 0) { "Green" } else { "Yellow" })
}

# Sjekk nyeste modeller
Write-Host ""
Write-Host "🔍 Sjekker modeller..." -ForegroundColor Yellow
$models = docker exec quantum_backend ls -t /app/ai_engine/models/*.pkl 2>$null | Select-Object -First 5

if ($models) {
    Write-Host "   Nyeste modeller:" -ForegroundColor White
    foreach ($model in $models) {
        $name = Split-Path $model -Leaf
        if ($name -match "v(\d{8}_\d{6})") {
            $timestamp = $Matches[1]
            Write-Host "   • $name" -ForegroundColor Green
        }
    }
}

# Sjekk om continuous training kjører
Write-Host ""
Write-Host "🔍 Sjekker training prosess..." -ForegroundColor Yellow
$logCheck = docker logs quantum_backend 2>&1 | Select-String "TRENING SUKSESS" | Select-Object -Last 1

if ($logCheck) {
    Write-Host "✅ Training kjører og fungerer!" -ForegroundColor Green
    Write-Host "   Siste suksess: $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor White
} else {
    Write-Host "⚠️  Ingen nylig training aktivitet" -ForegroundColor Yellow
    Write-Host "   Kjør: .\start_training.ps1" -ForegroundColor White
}

Write-Host ""
Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "💡 Tips:" -ForegroundColor Yellow
Write-Host "   • Training kjører hver 5. minutt automatisk" -ForegroundColor White
Write-Host "   • Sjekk logs med: docker logs quantum_backend | Select-String TRENING" -ForegroundColor White
Write-Host "   • Start training med: .\start_training.ps1" -ForegroundColor White
Write-Host ""
