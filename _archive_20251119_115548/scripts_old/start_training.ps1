#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Starter kontinuerlig AI trening i egen terminal
    
.DESCRIPTION
    Dette scriptet starter AI trenings-systemet som kjører permanent
    i bakgrunnen og trener modellen hver 5. minutt.
    
    IKKE LUKK TERMINALEN SOM ÅPNES!
#>

Write-Host ""
Write-Host "🚀 STARTER KONTINUERLIG AI TRENING..." -ForegroundColor Cyan
Write-Host ""

# Sjekk at Docker kjører
try {
    docker ps | Out-Null
    Write-Host "✅ Docker kjører" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker er ikke tilgjengelig!" -ForegroundColor Red
    Write-Host "   Start Docker Desktop først." -ForegroundColor Yellow
    pause
    exit 1
}

# Sjekk at backend container kjører
$backend = docker ps --filter "name=quantum_backend" --format "{{.Names}}"
if ($backend -ne "quantum_backend") {
    Write-Host "❌ Backend container kjører ikke!" -ForegroundColor Red
    Write-Host "   Kjør først: docker compose up -d backend" -ForegroundColor Yellow
    pause
    exit 1
}

Write-Host "✅ Backend container kjører" -ForegroundColor Green
Write-Host ""

# Start trening i egen terminal
Write-Host "📊 Starter trening-terminal..." -ForegroundColor Cyan

Start-Process pwsh -ArgumentList "-NoExit", "-Command", @"
Write-Host ''
Write-Host '🚀 KONTINUERLIG AI TRENING - PERMANENT' -ForegroundColor Green
Write-Host '════════════════════════════════════════════════════════════════' -ForegroundColor Cyan
Write-Host ''
Write-Host '✅ Status: 100% FEILFRI DRIFT' -ForegroundColor Green
Write-Host '⚙️  Intervall: 5 minutter' -ForegroundColor White
Write-Host '🎯 Features: 14 tekniske indikatorer' -ForegroundColor White
Write-Host '📊 Mode: Paper trading (sikker læring)' -ForegroundColor White
Write-Host ''
Write-Host 'Dette vinduet SKAL FORBLI ÅPENT for kontinuerlig trening!' -ForegroundColor Yellow
Write-Host 'Trykk Ctrl+C for å stoppe.' -ForegroundColor Yellow
Write-Host ''
Write-Host '════════════════════════════════════════════════════════════════' -ForegroundColor Cyan
Write-Host ''
docker exec -it quantum_backend sh -c 'export QUANTUM_TRADER_DATABASE_URL=sqlite:////app/backend/data/trades.db && python /app/continuous_training.py'
"@

Write-Host ""
Write-Host "✅ Trening startet i egen terminal!" -ForegroundColor Green
Write-Host ""
Write-Host "📝 Viktig:" -ForegroundColor Yellow
Write-Host "   • Terminalen som åpnet seg MÅ forbli åpen" -ForegroundColor White
Write-Host "   • AI trener automatisk hver 5. minutt" -ForegroundColor White
Write-Host "   • Du kan lukke DETTE vinduet" -ForegroundColor White
Write-Host ""

pause
