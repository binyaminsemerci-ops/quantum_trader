@echo off
setlocal enabledelayedexpansion

REM Resolve repo root (this script is in repo root)
set REPO_ROOT=%~dp0
set REPO_ROOT=%REPO_ROOT:~0,-1%

REM Paths
set VENV=%REPO_ROOT%\.venv
set PY=%VENV%\Scripts\python.exe
set TRAIN=%REPO_ROOT%\backend\train_model.py

REM Default data file (override by passing a path: train_ai_model.bat C:\path\data.csv)
set DATA=%REPO_ROOT%\data\your_data.csv
if not "%~1"=="" set DATA=%~1

if not exist "%PY%" (
  echo Python venv not found at %PY%
  echo Create it: python -m venv .venv
  exit /b 1
)

if not exist "%TRAIN%" (
  echo Train script not found: %TRAIN%
  exit /b 1
)

if not exist "%DATA%" (
  echo Data file not found: %DATA%
  echo Pass a CSV path: train_ai_model.bat C:\path\to\data.csv
  exit /b 1
)

echo Running training with data: %DATA%
"%PY%" "%TRAIN%" "%DATA%"
endlocal
