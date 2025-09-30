@echo off
setlocal

REM Change to repo root
pushd "%~dp0"

REM If a local virtual environment exists, activate it so python/uvicorn use those packages.
IF EXIST .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

REM Launch backend in a new window
start "QuantumTrader Backend" cmd /k "cd /d %cd% && python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000"

REM Launch frontend dev server in a new window
start "QuantumTrader Frontend" cmd /k "cd /d %cd%\frontend && npm run dev"

popd
endlocal
