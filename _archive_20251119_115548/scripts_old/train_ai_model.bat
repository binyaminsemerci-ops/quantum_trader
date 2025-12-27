@echo off
setlocal enabledelayedexpansion

REM Determine repository root (directory containing this script)
set "REPO_ROOT=%~dp0"
pushd "%REPO_ROOT%" >nul

REM Locate or bootstrap Python interpreter
set "PY_BOOTSTRAP="
if exist ".venv\Scripts\python.exe" (
    set "PY_BOOTSTRAP=.venv\Scripts\python.exe"
) else (
    for %%P in (py.exe python.exe) do (
        if not defined PY_BOOTSTRAP (
            where %%P >nul 2>&1
            if not errorlevel 1 (
                if /I "%%P"=="py.exe" (
                    set "PY_BOOTSTRAP=py -3"
                ) else (
                    set "PY_BOOTSTRAP=python"
                )
            )
        )
    )
    if not defined PY_BOOTSTRAP (
        echo [ERROR] Python 3 interpreter not found. Install Python 3.9+ and rerun.
        goto :fail
    )
    echo [INFO] Creating virtual environment in .venv ...
    %PY_BOOTSTRAP% -m venv .venv || goto :fail
)

set "VENV_PY=%REPO_ROOT%.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
    echo [ERROR] Unable to locate Python interpreter at "%VENV_PY%".
    goto :fail
)

REM Install dependencies the first time (skip stdlib sqlite3 entry)
set "BOOTSTRAP_MARKER=.venv\.trained_env_ready"
if not exist "%BOOTSTRAP_MARKER%" (
    echo [INFO] Installing Python dependencies ...
    "%VENV_PY%" -m pip install --upgrade pip || goto :fail

    if exist "requirements.txt" (
        set "TMP_REQ=%TEMP%\qt_req_main_!RANDOM!.txt"
        "%VENV_PY%" -c "from pathlib import Path; src=Path(r'requirements.txt'); tmp=Path(r'!TMP_REQ!'); helper=lambda s: (lambda head: head and head.startswith('sqlite3'))(s.split('#',1)[0].strip().lower()); filtered=[line for line in src.read_text().splitlines() if not helper(line)]; tmp.write_text('\n'.join(filtered)+('\n' if filtered else ''), encoding='utf-8')" || goto :fail
        "%VENV_PY%" -m pip install -r "!TMP_REQ!" || goto :fail
        del "!TMP_REQ!" >nul 2>&1
    )

    if exist "backend\requirements.txt" (
        set "TMP_REQ=%TEMP%\qt_req_backend_!RANDOM!.txt"
        "%VENV_PY%" -c "from pathlib import Path; src=Path(r'backend\requirements.txt'); tmp=Path(r'!TMP_REQ!'); helper=lambda s: (lambda head: head and head.startswith('sqlite3'))(s.split('#',1)[0].strip().lower()); filtered=[line for line in src.read_text().splitlines() if not helper(line)]; tmp.write_text('\n'.join(filtered)+('\n' if filtered else ''), encoding='utf-8')" || goto :fail
        "%VENV_PY%" -m pip install -r "!TMP_REQ!" || goto :fail
        del "!TMP_REQ!" >nul 2>&1
    )

    >"%BOOTSTRAP_MARKER%" echo ready
)

REM Ensure model output directory exists
if not exist "ai_engine\models" (
    mkdir "ai_engine\models" >nul 2>&1
)

if not exist "train_ai.py" (
    echo [ERROR] Could not find train_ai.py in "%REPO_ROOT%".
    goto :fail
)

REM Run training script (train_ai.py uses public Binance/CoinGecko data by default)
echo [INFO] Starting AI model training ...
"%VENV_PY%" train_ai.py %*
set "ERR=%ERRORLEVEL%"

if "%ERR%"=="0" (
    echo [INFO] Training complete. Check ai_engine\models\ for artifacts.
) else (
    echo [ERROR] Training failed with exit code %ERR%.
)

popd >nul
exit /b %ERR%

:fail
popd >nul
exit /b 1

