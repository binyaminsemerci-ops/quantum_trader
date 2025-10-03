@echo off
REM Quantum Trader Orchestrator - One unified start / maintenance script
REM Usage: Double-click or run in terminal. Choose option.
REM Safe defaults: will NOT delete node_modules or virtualenv unless you confirm.

set SCRIPT_DIR=%~dp0
cd /d %SCRIPT_DIR%

set VENV_DIR=.venv
set PYTHON_EXE=%VENV_DIR%\Scripts\python.exe
set UVICORN_CMD=uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
set FRONTEND_DIR=frontend

:MENU
echo(
echo ================= Quantum Trader Orchestrator =================
echo   A) Clean      - Purge caches/logs/temp files
echo   B) Install    - Install all dependencies  
echo   C) Build      - Build and validate project
echo   D) Fix        - Auto-fix code issues
echo   E) Test       - Run all tests
echo   F) Start      - Launch development servers
echo   G) Train      - AI model training
echo   Q) Quit       - Exit orchestrator
echo ===============================================================
set /p CHOICE=Select option (A-G,Q): 

if /I "%CHOICE%"=="A" goto CLEAN
if /I "%CHOICE%"=="B" goto INSTALL
if /I "%CHOICE%"=="C" goto BUILD
if /I "%CHOICE%"=="D" goto AUTOFIX
if /I "%CHOICE%"=="E" goto TEST_ALL
if /I "%CHOICE%"=="F" goto START
if /I "%CHOICE%"=="G" goto TRAIN
if /I "%CHOICE%"=="Q" goto END
echo Invalid choice.
goto MENU

:ENSURE_VENV
if exist "%PYTHON_EXE%" goto :eof
echo [INFO] Creating virtual environment...
py -3 -m venv %VENV_DIR%
if not exist "%PYTHON_EXE%" (
  echo [ERROR] Failed to create virtual environment.
  goto MENU
)
%PYTHON_EXE% -m pip install --upgrade pip >nul 2>&1
goto :eof

:CLEAN
echo [CLEAN] Removing Python caches, logs, artifacts, SQLite DBs.
for /d /r %%D in (__pycache__) do rd /s /q "%%D" 2>nul
del /s /q *.pyc *.pyo *.pyd 2>nul
if exist .pytest_cache rd /s /q .pytest_cache
if exist .ruff_cache rd /s /q .ruff_cache
if exist artifacts rd /s /q artifacts
if exist backend\data\trades.db del /q backend\data\trades.db
if exist backend\quantum_trader.db del /q backend\quantum_trader.db
if exist trades.db del /q trades.db
if exist db.sqlite3 del /q db.sqlite3
if exist logs rd /s /q logs
mkdir logs >nul 2>&1
echo [CLEAN] Base cleanup complete.
set /p DELNODE=Delete node_modules (y/N)? : 
if /I "%DELNODE%"=="Y" (
  if exist %FRONTEND_DIR%\node_modules rd /s /q %FRONTEND_DIR%\node_modules
  echo [CLEAN] node_modules removed.
)
set /p DELVENV=Delete virtualenv .venv (y/N)? : 
if /I "%DELVENV%"=="Y" (
  if exist %VENV_DIR% rd /s /q %VENV_DIR%
  echo [CLEAN] Virtualenv removed.
)
echo [CLEAN] Done.
goto MENU

:INSTALL
call :ENSURE_VENV
echo [BOOTSTRAP] Installing backend requirements...
%PYTHON_EXE% -m pip install -r requirements.txt || goto BOOT_FAIL
if exist backend\requirements.txt %PYTHON_EXE% -m pip install -r backend\requirements.txt >nul 2>&1
if exist backend\requirements-dev.txt %PYTHON_EXE% -m pip install -r backend\requirements-dev.txt >nul 2>&1
echo [BOOTSTRAP] Backend deps OK.
if exist %FRONTEND_DIR%\package.json (
  echo [BOOTSTRAP] Installing frontend dependencies (npm ci || npm install)...
  pushd %FRONTEND_DIR%
  call npm ci 2>nul || call npm install || goto BOOT_FAIL
  popd
  echo [BOOTSTRAP] Frontend deps OK.
) else (
  echo [WARN] No frontend package.json found.
)
echo [BOOTSTRAP] Completed.
goto MENU

:BOOT_FAIL
echo [ERROR] Bootstrap failed.
goto MENU

:BUILD
call :ENSURE_VENV
echo [BUILD] Building and validating project components...
echo [BUILD] Backend validation...
%PYTHON_EXE% -c "import backend.main, backend.simple_main; print('Backend imports OK')" 2>nul
if errorlevel 1 (
  echo [BUILD] Backend import failed. Run B (Install) first.
  pause
  goto MENU
)
echo [BUILD] Backend validation passed.
if exist %FRONTEND_DIR%\package.json (
  echo [BUILD] Frontend build...
  cd /d %FRONTEND_DIR%
  call npm run build
  if errorlevel 1 (
    echo [BUILD] Frontend build failed. Run B (Install) first.
    cd /d %SCRIPT_DIR%
    pause
    goto MENU
  )
  echo [BUILD] Frontend build successful.
  cd /d %SCRIPT_DIR%
)
echo [BUILD] Complete - project is ready.
pause
goto MENU

:START
call :ENSURE_VENV
echo [START] Launching backend and frontend in separate windows.
start "QT-Backend" cmd /k "cd /d %SCRIPT_DIR% & call %VENV_DIR%\Scripts\activate && %UVICORN_CMD%"
if exist %FRONTEND_DIR%\package.json (
  start "QT-Frontend" cmd /k "cd /d %SCRIPT_DIR%%FRONTEND_DIR% && npm run dev"
) else (
  echo [START] Frontend directory missing or no package.json
)
echo [START] Services launched.
goto MENU

:TEST_ALL
call :ENSURE_VENV
REM --- Auto minimal bootstrap if core libs missing ---
"%PYTHON_EXE%" -c "import fastapi" >nul 2>&1
if errorlevel 1 (
  echo [TEST] fastapi missing - performing minimal dependency install...
  for /f %%m in ('"%PYTHON_EXE%" -c "import sys;print(sys.version_info.minor)"') do set PYMIN=%%m
  if exist backend\requirements.txt (
    if %PYMIN% GEQ 13 (
      echo [TEST] Python .%%PYMIN%% detected (>=3.13) - filtering heavy ML deps (scikit-learn, xgboost)
      findstr /R /I /V "scikit-learn xgboost" backend\requirements.txt > backend\requirements-lite-tmp.txt
      "%PYTHON_EXE%" -m pip install -r backend\requirements-lite-tmp.txt || echo [TEST] Lite install failed.
    ) else (
      "%PYTHON_EXE%" -m pip install -r backend\requirements.txt || echo [TEST] Install failed.
    )
  ) else (
    echo [TEST] backend/requirements.txt not found, falling back to root requirements.txt
    "%PYTHON_EXE%" -m pip install -r requirements.txt >nul 2>&1
  )
)
echo [TEST] Running backend pytest suite...
%PYTHON_EXE% -m pytest -q
if errorlevel 1 (
  echo [TEST] Backend tests FAILED.
  echo.
  echo [HINT] Hvis du bruker Python 3.13 og ser mange build-feil: sett ^"QT_PY_VERSION=3.12^", slett .venv og kjør B (Bootstrap) igjen.
  echo [HINT] Du kan også midlertidig deaktivere tunge libs: set USE_XGB=0 og set USE_SKLEARN=0
  if not defined QT_NONINTERACTIVE pause
) else (
  echo [TEST] Backend tests passed.
)
if exist %FRONTEND_DIR%\package.json (
  echo [TEST] Building frontend (npm run build)...
  pushd %FRONTEND_DIR%
  call npm run build
  if errorlevel 1 (
    echo [TEST] Frontend build FAILED.
  ) else (
    echo [TEST] Frontend build OK.
  )
  popd
)
echo [TEST] Complete.
if not defined QT_NONINTERACTIVE pause >nul
goto MENU

:AUTOFIX
call :ENSURE_VENV
echo [AUTOFIX] Running ruff (if installed)...
%PYTHON_EXE% -m ruff check --fix . 2>nul
echo [AUTOFIX] Ensuring dependencies present...
%PYTHON_EXE% -m pip install -r requirements.txt >nul 2>&1
if exist backend\requirements-dev.txt %PYTHON_EXE% -m pip install -r backend\requirements-dev.txt >nul 2>&1
echo [AUTOFIX] Quick import smoke check...
%PYTHON_EXE% - <<PYEOF 2>nul
try:
    import backend.main, backend.simple_main
    print('IMPORT_OK')
except Exception as e:
    print('IMPORT_FAIL', e)
PYEOF
echo [AUTOFIX] Done.
goto MENU

:TRAIN
call :ENSURE_VENV
REM Determine limit (env override QT_TRAIN_LIMIT)
set LIMIT=1200
if defined QT_TRAIN_LIMIT set LIMIT=%QT_TRAIN_LIMIT%
echo [TRAIN] Starting AI training (limit=%LIMIT%, backtest disabled)...

REM Build inline Python command (keep it one line for -c)
set "PYCMD=from ai_engine.train_and_save import train_and_save; r=train_and_save(limit=%LIMIT%, backtest=False, write_report=True); print('METRICS', r.get('metrics')); print('MODEL', r.get('model_path'))"
"%PYTHON_EXE%" -c "%PYCMD%"
if errorlevel 1 (
  echo [TRAIN] FAILED (exit code %errorlevel%).
  echo [HINT] Mulig tung avhengighet feilet. Du kan teste med: set USE_XGB=0 og set USE_SKLEARN=0
  echo [HINT] Eller reduser datamengde: set QT_TRAIN_LIMIT=200
  if not defined QT_NONINTERACTIVE pause
) else (
  echo [TRAIN] Completed.
  if not defined QT_NONINTERACTIVE pause
)
goto MENU

:END
echo Exiting orchestrator.
exit /b 0