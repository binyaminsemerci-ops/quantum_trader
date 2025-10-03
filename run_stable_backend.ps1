# Stable backend runner - no reload, enhanced logging
# Usage: .\run_stable_backend.ps1

Write-Host "Starting Quantum Trader Backend (Stable Mode)" -ForegroundColor Green
Write-Host "Heartbeat logging every 15s, watchlist caching enabled" -ForegroundColor Yellow

# Run uvicorn without reload for maximum stability
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --log-level info --no-access-log

Write-Host "Backend stopped" -ForegroundColor Red