<#!
.SYNOPSIS
  Full clean rebuild of Quantum Trader (backend + frontend) on Windows PowerShell.
.DESCRIPTION
  Removes caches, node_modules, dist artifacts, Python venv (optional), SQLite DBs, Alembic versions (optional), then reinstalls + migrates.
.PARAMETER PreserveVenv
  Keep existing .venv if present.
.PARAMETER PreserveDB
  Keep existing SQLite databases and migration history.
.PARAMETER Fast
  Skip dependency reinstalls if folders exist.
.EXAMPLE
  ./scripts/full-reset.ps1
.EXAMPLE
  ./scripts/full-reset.ps1 -PreserveVenv -PreserveDB
#>
param(
  [switch]$PreserveVenv,
  [switch]$PreserveDB,
  [switch]$Fast
)

$ErrorActionPreference = 'Stop'
function Section($t){ Write-Host "`n=== $t ===" -ForegroundColor Cyan }
function Info($t){ Write-Host "[info] $t" -ForegroundColor Gray }
function Warn($t){ Write-Host "[warn] $t" -ForegroundColor Yellow }
function Done($t){ Write-Host "[done] $t" -ForegroundColor Green }

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $root '..')
Push-Location $repoRoot

Section 'KILL RUNNING PROCESSES'
Get-Process node -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "*quantum_trader*" } | Stop-Process -Force -ErrorAction SilentlyContinue
Done 'Processes terminated (if any)'

Section 'BACKEND CLEAN'
if (-not $PreserveDB){
  Get-ChildItem backend -Include trades.db,*.sqlite3 -Recurse -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
  if (Test-Path backend/data){ Get-ChildItem backend/data -Include *.db -Recurse | Remove-Item -Force -ErrorAction SilentlyContinue }
  Info 'Removed SQLite db files'
}
else { Warn 'Preserving DB files' }

if (-not $Fast){
  if (-not $PreserveVenv -and (Test-Path .venv)) { Remove-Item .venv -Recurse -Force; Info 'Removed existing .venv' }
  if (-not $PreserveVenv){ python -m venv .venv; Info 'Created new venv' }
  . .\.venv\Scripts\Activate.ps1
  python -m pip install --upgrade pip
  pip install -r backend/requirements.txt
  if (Test-Path backend/requirements-dev.txt){ pip install -r backend/requirements-dev.txt }
  Done 'Python deps installed'
} else { Warn 'Fast mode: skipping backend dependency reinstall' }

if (-not $PreserveDB){
  if (Test-Path backend/migrations){ alembic -c backend/alembic.ini upgrade head }
  Done 'Applied migrations'
}

Section 'FRONTEND CLEAN'
if (Test-Path frontend/node_modules -and -not $Fast){ Remove-Item frontend/node_modules -Recurse -Force; Info 'Deleted node_modules' } elseif ($Fast){ Warn 'Fast mode: keeping node_modules' }
Get-ChildItem frontend -Include dist,.turbo -Directory -Recurse -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem frontend -Include vite*.log -Recurse -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
if (-not $Fast){
  Push-Location frontend
  npm install
  Pop-Location
  Done 'Frontend deps installed'
}

Section 'VERIFY TAILWIND CONFIG'
if (-not (Test-Path frontend/tailwind.config.ts)){ Warn 'tailwind.config.ts missing' } else { Info 'tailwind.config.ts present' }
if (Test-Path frontend/tailwind.config.tsx){ Warn 'Found deprecated tailwind.config.tsx (should be removed)' }
if (-not (Test-Path frontend/postcss.config.js)){ Warn 'postcss.config.js missing' }

Section 'SMOKE BUILD FRONTEND'
Push-Location frontend
npx tailwindcss -i ./src/index.css -o ./tailwind-debug.css --minify
if (-not (Select-String -Path ./tailwind-debug.css -Pattern ".dark:" -Quiet)) { Warn 'Dark variants not found in output CSS -> check content paths' } else { Info 'Dark variants detected in tailwind-debug.css' }
Pop-Location

Section 'START BACKEND'
Start-Process powershell -ArgumentList '-NoExit','-Command','cd $PWD/backend; . ../../.venv/Scripts/Activate.ps1; uvicorn main:app --reload --port 8000' | Out-Null
Info 'Backend starting in separate window'

Section 'START FRONTEND'
Start-Process powershell -ArgumentList '-NoExit','-Command','cd $PWD/frontend; npm run dev' | Out-Null
Info 'Frontend starting in separate window'

Section 'SUMMARY'
Write-Host "Dark mode test: open http://localhost:5173 and run in console: document.documentElement.classList.add('dark')" -ForegroundColor Magenta
Write-Host "Compact mode test: run document.documentElement.classList.add('compact-mode')" -ForegroundColor Magenta

Done 'Full reset complete'

Pop-Location
