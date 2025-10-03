<#
 Quantum Trader Orchestrator (PowerShell)
 Logical development workflow:
 A Clean → B Install → C Build → D Fix → E Test → F Start → G Train
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repo
$venv = Join-Path $repo '.venv'
$python = Join-Path $venv 'Scripts/python.exe'

function Ensure-Venv {
  if (Test-Path $python) { return }
  Write-Host '[INFO] Creating venv' -ForegroundColor Cyan
  $pySpec = $Env:QT_PY_VERSION
  $created = $false
  if ($pySpec) {
    Write-Host "[INFO] Using QT_PY_VERSION=$pySpec" -ForegroundColor DarkCyan
    try { py -$pySpec -m venv $venv 2>$null; $created = $true } catch { Write-Host "[WARN] Could not use py -$pySpec" -ForegroundColor Yellow }
  }
  if (-not $created) {
    foreach ($candidate in @('3.12','3','')) {
      if ($created) { break }
      try {
        if ($candidate -eq '') { py -3 -m venv $venv 2>$null } else { py -$candidate -m venv $venv 2>$null }
        if (Test-Path $python) { $created = $true }
      } catch { }
    }
  }
  if (-not $created -and -not (Test-Path $python)) {
    Write-Host '[WARN] Falling back to system python' -ForegroundColor Yellow
    python -m venv $venv
  }
  if (!(Test-Path $python)) { throw 'Failed to create virtualenv' }
  & $python -m pip install --upgrade pip | Out-Null
  # Python-version aware numpy compatibility (handle fresh Python 3.13 environments)
  $pyVer = & $python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
  if ($pyVer -ge '3.13') {
  Write-Host "[INFO] Detected Python ${pyVer}: ensuring modern numpy wheel" -ForegroundColor DarkCyan
    & $python -m pip install --upgrade "numpy>=2.1.0" 2>$null
    Write-Host '[INFO] Python >=3.13 detected. If pandas/xgboost wheels missing: set QT_PY_VERSION=3.12 and recreate venv.' -ForegroundColor DarkYellow
  }
}

function Do-Clean {
  Write-Host '[CLEAN] Purging caches/logs/sqlite dbs' -ForegroundColor Yellow
  Get-ChildItem -Recurse -Force -Include '__pycache__' | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
  Get-ChildItem -Recurse -Force -Include '*.pyc','*.pyo' | Remove-Item -Force -ErrorAction SilentlyContinue
  Remove-Item backend\data\trades.db, backend\quantum_trader.db, trades.db, db.sqlite3 -ErrorAction SilentlyContinue
  if (Test-Path logs) { Remove-Item logs -Recurse -Force }
  New-Item -ItemType Directory -Path logs | Out-Null
  $nonInteractive = $Env:QT_NONINTERACTIVE -eq '1'
  if (-not $nonInteractive) {
    if ((Read-Host 'Delete node_modules? (y/N)') -match '^[Yy]') { if (Test-Path frontend\node_modules) { Remove-Item frontend\node_modules -Recurse -Force } }
    if ((Read-Host 'Delete .venv? (y/N)') -match '^[Yy]') { if (Test-Path $venv) { Remove-Item $venv -Recurse -Force } }
  } else {
    Write-Host '[CLEAN] Non-interactive mode: skipping node_modules/.venv deletion' -ForegroundColor DarkGray
  }
  Write-Host '[CLEAN] Done' -ForegroundColor Green
}

function Invoke-PipInstall($file) {
  if (!(Test-Path $file)) { return }
  Write-Host "  -> $file" -ForegroundColor DarkCyan
  & $python -m pip install -r $file
  if ($LASTEXITCODE -ne 0) { throw "pip install failed for $file" }
}

function Do-Install {
  Ensure-Venv
  Write-Host '[INSTALL] Python dependencies' -ForegroundColor Cyan
  # Install root (legacy) first only if backend pinned files absent
  if (!(Test-Path backend\requirements.txt)) { Invoke-PipInstall 'requirements.txt' }
  # Always install pinned backend requirements if present (runtime then dev)
  Invoke-PipInstall 'backend/requirements.txt'
  Invoke-PipInstall 'backend/requirements-dev.txt'
  # Frontend
  if (Test-Path frontend\package.json) {
    Write-Host '[INSTALL] Frontend dependencies' -ForegroundColor Cyan
    Push-Location frontend
    try {
      # Try npm ci first (faster for CI/lockfile)
      Write-Host '[INSTALL] Running npm ci...' -ForegroundColor DarkCyan
      $npmOut = npm ci 2>&1
      if ($LASTEXITCODE -ne 0) { 
        Write-Host '[INSTALL] npm ci failed, trying npm install...' -ForegroundColor Yellow
        $npmOut = npm install 2>&1
      }
      if ($LASTEXITCODE -ne 0) { 
        Write-Host '[INSTALL] Frontend npm failed:' -ForegroundColor Red
        if ($npmOut) { $npmOut | ForEach-Object { Write-Host "  $_" -ForegroundColor Red } }
        Pop-Location
        if (-not ($Env:QT_NONINTERACTIVE -eq '1')) { Read-Host 'Press ENTER to continue' | Out-Null }
        return
      }
      Write-Host '[INSTALL] Frontend npm install complete' -ForegroundColor Green
      
      # Ensure vitest is available (VS Code extension expects it)
      Write-Host '[INSTALL] Checking vitest...' -ForegroundColor DarkCyan
      npm list vitest 2>$null
      if ($LASTEXITCODE -ne 0) {
        Write-Host '[INSTALL] Installing vitest for VS Code compatibility' -ForegroundColor Yellow
        $vitestInstall = npm install --save-dev vitest 2>&1
        if ($LASTEXITCODE -ne 0) {
          Write-Host '[INSTALL] Vitest install failed (continuing anyway)' -ForegroundColor Yellow
          if ($vitestInstall) { $vitestInstall | ForEach-Object { Write-Host "  $_" } }
        }
      }
    } finally {
      Pop-Location
    }
  }
  Write-Host '[INSTALL] Complete' -ForegroundColor Green
}

function Do-Build {
  Ensure-Venv
  Write-Host '[BUILD] Building project components' -ForegroundColor Cyan
  
  # Build backend (ensure it imports correctly)
  Write-Host '[BUILD] Backend validation' -ForegroundColor Yellow
  $importOut = & $python -c "import backend.main, backend.simple_main; print('Backend imports OK')" 2>&1
  $importCode = $LASTEXITCODE
  if ($importCode -ne 0) {
    Write-Host '[BUILD] Backend import failed:' -ForegroundColor Red
    if ($importOut) { $importOut | ForEach-Object { Write-Host "[BUILD-OUT] $_" } }
    Write-Host 'Hint: Run B (Install) first to ensure all dependencies are available.' -ForegroundColor DarkYellow
    if (-not ($Env:QT_NONINTERACTIVE -eq '1')) { Read-Host 'Press ENTER to return to menu' | Out-Null }
    return
  }
  Write-Host '[BUILD] Backend validation passed' -ForegroundColor Green
  
  # Build frontend if exists
  if (Test-Path frontend\package.json) {
    Write-Host '[BUILD] Frontend build' -ForegroundColor Yellow
    Push-Location frontend
    try {
      $buildOut = npm run build 2>&1
      $buildCode = $LASTEXITCODE
      if ($buildOut) { $buildOut | ForEach-Object { Write-Host "[BUILD-OUT] $_" } }
      if ($buildCode -ne 0) {
        Write-Host '[BUILD] Frontend build failed' -ForegroundColor Red
        Write-Host 'Hint: Run B (Install) first to ensure frontend dependencies are installed.' -ForegroundColor DarkYellow
        if (-not ($Env:QT_NONINTERACTIVE -eq '1')) { Read-Host 'Press ENTER to return to menu' | Out-Null }
        return
      }
      Write-Host '[BUILD] Frontend build successful' -ForegroundColor Green
    } finally {
      Pop-Location
    }
  }
  Write-Host '[BUILD] Complete - project is ready' -ForegroundColor Green
}

function Do-Start {
  Ensure-Venv
  Write-Host 'Tip 1: Set env QT_PY_VERSION=3.12 then delete .venv to use Python 3.12 wheels.' -ForegroundColor DarkYellow
  Write-Host 'Tip 2: Disable heavy libs: set USE_XGB=0 USE_SKLEARN=0 to force fallback models.' -ForegroundColor DarkYellow
  Write-Host '[START] Launching dev servers' -ForegroundColor Cyan
  if (Test-Path frontend\package.json) {
    Start-Process powershell -ArgumentList "-NoExit","-Command","Set-Location '$repo/frontend'; npm run dev" -WorkingDirectory "$repo/frontend"
  }
  Write-Host '[START] Launched' -ForegroundColor Green
}

function Do-Test {
  Ensure-Venv
  # Quick sanity: ensure core deps installed (in case user skipped B)
  if (!(Get-Command "$python" -ErrorAction SilentlyContinue)) { throw 'Python venv missing' }
  $needed = @('pytest','fastapi','sqlalchemy')
  $missing = @()
  foreach ($n in $needed) {
    & $python -c "import importlib,sys; import $n" 2>$null; if ($LASTEXITCODE -ne 0) { $missing += $n }
  }
  if ($missing.Count -gt 0) {
    Write-Host "[TEST] Missing deps: $($missing -join ', ')" -ForegroundColor Yellow
    if (Test-Path backend/requirements.txt) {
      Write-Host '[TEST] Running targeted bootstrap for backend requirements' -ForegroundColor DarkYellow
      & $python -m pip install -r backend/requirements.txt
      & $python -m pip install -r backend/requirements-dev.txt 2>$null
    } else {
      & $python -m pip install $missing
    }
  }
  Write-Host '[TEST] Backend pytest' -ForegroundColor Cyan
  # Check if tests exist first
  $testFiles = Get-ChildItem -Recurse -Path backend,tests -Include "*.py" -Name "*test*.py" -ErrorAction SilentlyContinue
  if ($testFiles.Count -eq 0) {
    Write-Host '[TEST] No test files found in backend/ or tests/ directories' -ForegroundColor Yellow
    Write-Host '[TEST] Skipping pytest (no tests to run)' -ForegroundColor DarkGray
  } else {
    $testOut = & $python -m pytest backend tests -v --tb=short 2>&1
    $testCode = $LASTEXITCODE
    if ($testOut) { $testOut | ForEach-Object { Write-Host "[TEST-OUT] $_" } }
    if ($testCode -ne 0) {
      Write-Host '[TEST] Pytest FAILED.' -ForegroundColor Red
      Write-Host 'Hint: If Python 3.13 wheels unstable, set QT_PY_VERSION=3.12 then delete .venv and run B.' -ForegroundColor DarkYellow
      if (-not ($Env:QT_NONINTERACTIVE -eq '1')) { Read-Host 'Press ENTER to return to menu' | Out-Null }
      return
    } else {
      Write-Host '[TEST] Backend tests passed' -ForegroundColor Green
    }
  }
  if (Test-Path frontend\package.json) {
    Push-Location frontend
    Write-Host '[TEST] Frontend build' -ForegroundColor Cyan
    $fbOut = npm run build 2>&1
    $fbCode = $LASTEXITCODE
    if ($fbOut) { $fbOut | ForEach-Object { Write-Host "[TEST-OUT] $_" } }
    if ($fbCode -ne 0) { 
      Pop-Location
      Write-Host '[TEST] Frontend build FAILED.' -ForegroundColor Red
      Write-Host 'Hint: Run B (Bootstrap) first to ensure all frontend deps installed.' -ForegroundColor DarkYellow
      if (-not ($Env:QT_NONINTERACTIVE -eq '1')) { Read-Host 'Press ENTER to return to menu' | Out-Null }
      return 
    }
    # Optional frontend tests (if vitest available)
    npm list vitest 2>$null; $vitestAvailable = ($LASTEXITCODE -eq 0)
    if ($vitestAvailable) {
      Write-Host '[TEST] Frontend vitest' -ForegroundColor Cyan
      $vtOut = npm run test 2>&1
      $vtCode = $LASTEXITCODE
      if ($vtOut) { $vtOut | ForEach-Object { Write-Host "[TEST-OUT] $_" } }
      if ($vtCode -ne 0) {
        Write-Host '[TEST] Frontend tests had issues (continuing anyway)' -ForegroundColor Yellow
      }
    } else {
      Write-Host '[TEST] Vitest not available, skipping frontend tests' -ForegroundColor DarkGray
    }
    Pop-Location
  }
  Write-Host '[TEST] Complete' -ForegroundColor Green
}

function Do-Autofix {
  Ensure-Venv
  Write-Host '[AUTOFIX] Ruff + deps + import smoke' -ForegroundColor Cyan
  $haveRuff = $false
  & $python -c "import ruff" 2>$null; if ($LASTEXITCODE -eq 0) { $haveRuff = $true }
  if (-not $haveRuff) {
    Write-Host '[AUTOFIX] Installing ruff (missing)' -ForegroundColor Yellow
    $ruffOut = & $python -m pip install ruff 2>&1
    if ($LASTEXITCODE -ne 0) {
      Write-Host '[AUTOFIX] Ruff install failed:' -ForegroundColor Red
      if ($ruffOut) { $ruffOut | ForEach-Object { Write-Host "  $_" } }
      if (-not ($Env:QT_NONINTERACTIVE -eq '1')) { Read-Host 'Press ENTER to return to menu' | Out-Null }
      return
    }
  }
  Write-Host '[AUTOFIX] Running ruff check/fix...' -ForegroundColor Cyan
  $ruffFixOut = & $python -m ruff check --fix backend ai_engine config 2>&1
  if ($ruffFixOut) { $ruffFixOut | ForEach-Object { Write-Host "[AUTOFIX-OUT] $_" } }
  
  Write-Host '[AUTOFIX] Ensuring deps...' -ForegroundColor Cyan
  if (Test-Path backend/requirements.txt) { & $python -m pip install -r backend/requirements.txt 2>$null }
  if (Test-Path backend/requirements-dev.txt) { & $python -m pip install -r backend/requirements-dev.txt 2>$null }
  
  Write-Host '[AUTOFIX] Import smoke test...' -ForegroundColor Cyan
  $smokeOut = & $python -c "import sys;print('[AUTOFIX] Import smoke:')\ntry:\n import backend.main, backend.simple_main; print('  IMPORT_OK')\nexcept Exception as e:\n print('  IMPORT_FAIL', e)" 2>&1
  if ($smokeOut) { $smokeOut | ForEach-Object { Write-Host "[AUTOFIX-OUT] $_" } }
  
  Write-Host '[AUTOFIX] Done' -ForegroundColor Green
}

function Do-Train {
  Ensure-Venv
  $limit = if ($Env:QT_TRAIN_LIMIT) { [int]$Env:QT_TRAIN_LIMIT } else { 1200 }
  Write-Host "[TRAIN] limit=$limit" -ForegroundColor Cyan
  # Guard for ai_engine deps (numpy, pandas, xgboost might be heavy; rely on backend pinned list)
  $coreML = @('numpy','pandas')
  $missingML = @()
  foreach ($m in $coreML) { & $python -c "import $m" 2>$null; if ($LASTEXITCODE -ne 0) { $missingML += $m } }
  if ($missingML.Count -gt 0) {
    Write-Host "[TRAIN] Installing missing ML deps: $($missingML -join ', ')" -ForegroundColor Yellow
    if (Test-Path backend/requirements.txt) { & $python -m pip install -r backend/requirements.txt }
  }
  Write-Host '[TRAIN] Running training...' -ForegroundColor Cyan
  # Test import first
  $importOut = & $python -c "from ai_engine.train_and_save import train_and_save; print('Import OK')" 2>&1
  $importCode = $LASTEXITCODE
  if ($importCode -ne 0) {
    Write-Host '[TRAIN] Import failed:' -ForegroundColor Red
    if ($importOut) { $importOut | ForEach-Object { Write-Host "[TRAIN-OUT] $_" } }
    Write-Host 'Tip: Run B (Bootstrap) first, or set USE_XGB=0 USE_SKLEARN=0 for fallback models.' -ForegroundColor DarkYellow
    if (-not ($Env:QT_NONINTERACTIVE -eq '1')) { Read-Host 'Press ENTER to return to menu' | Out-Null }
    return
  }
  # Run actual training
  $trainCmd = "from ai_engine.train_and_save import train_and_save; r=train_and_save(limit=$limit, backtest=False, write_report=True); print('Metrics:', r.get('metrics')); print('Model:', r.get('model_path'))"
  $out = & $python -c $trainCmd 2>&1
  $ec = $LASTEXITCODE
  if ($out) { $out | ForEach-Object { Write-Host "[TRAIN-OUT] $_" } }
  if ($ec -ne 0) {
    Write-Host '[TRAIN] Failed.' -ForegroundColor Red
    Write-Host 'Tip: set USE_XGB=0 USE_SKLEARN=0 (fallback models) or QT_PY_VERSION=3.12 then recreate venv.' -ForegroundColor DarkYellow
    if (-not ($Env:QT_NONINTERACTIVE -eq '1')) { Read-Host 'Press ENTER to return to menu' | Out-Null }
    return
  }
  Write-Host '[TRAIN] Complete' -ForegroundColor Green
}

function Menu {
  Write-Host ''
  Write-Host '=== Quantum Trader Orchestrator (PS) ===' -ForegroundColor Magenta
  Write-Host ' A Clean      - Purge caches/logs/temp files' -ForegroundColor DarkGray
  Write-Host ' B Install    - Install all dependencies' -ForegroundColor DarkCyan
  Write-Host ' C Build      - Build and validate project' -ForegroundColor Yellow
  Write-Host ' D Fix        - Auto-fix code issues' -ForegroundColor DarkYellow
  Write-Host ' E Test       - Run all tests' -ForegroundColor Cyan
  Write-Host ' F Start      - Launch development servers' -ForegroundColor Green
  Write-Host ' G Train      - AI model training' -ForegroundColor Magenta
  Write-Host ' Q Quit       - Exit orchestrator' -ForegroundColor Red
}

do {
  Menu
  $c = Read-Host 'Choose (A-G,Q)'
  try {
    switch ($c.ToUpper()) {
      'A' { Do-Clean }
      'B' { Do-Install }
      'C' { Do-Build }
      'D' { Do-Autofix }
      'E' { Do-Test }
      'F' { Do-Start }
      'G' { Do-Train }
      'Q' { Write-Host 'Exiting...' -ForegroundColor Yellow }
      Default { Write-Host 'Invalid choice' -ForegroundColor Red }
    }
  } catch {
    Write-Host ("[ERROR] $_") -ForegroundColor Red
    if (-not ($Env:QT_NONINTERACTIVE -eq '1')) { Read-Host 'Press ENTER to continue' | Out-Null }
  }
} while ($c.ToUpper() -ne 'Q')

if (-not ($Env:QT_NONINTERACTIVE -eq '1')) { Read-Host 'Press ENTER to exit' | Out-Null }
return