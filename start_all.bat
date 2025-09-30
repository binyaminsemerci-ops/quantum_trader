@echo off
setlocal

REM Change to repo root
pushd "%~dp0"

echo Starting Postgres (docker compose up -d db)...
docker compose up -d db
IF ERRORLEVEL 1 (
    echo Failed to start docker compose service 'db'.
    echo Check Docker Desktop/daemon and try again.
)

IF EXIST .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

start "QuantumTrader Backend" cmd /k "cd /d %cd% && python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000"
start "QuantumTrader Frontend" cmd /k "cd /d %cd%\frontend && npm run dev"
start "QuantumTrader Scheduler" cmd /k "cd /d %cd% && python scripts/scheduler.py"
start "QuantumTrader Autotrader" cmd /k "cd /d %cd% && python scripts/autotrader.py"

popd
endlocal
