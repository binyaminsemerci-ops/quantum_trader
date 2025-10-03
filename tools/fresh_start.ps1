<#!
Fresh Start Orchestrator

Usage examples (fra repo rot):
  pwsh -File .\tools\fresh_start.ps1               # Normal: backend + prod build + static serve
  pwsh -File .\tools\fresh_start.ps1 -ResetDb      # Sletter SQLite DB først
  pwsh -File .\tools\fresh_start.ps1 -OnlyBackend  # Bare backend (ingen build frontend)
  pwsh -File .\tools\fresh_start.ps1 -OnlyFrontend # Bare prod-build + static server

Flags kan kombineres der det gir mening.
!#>
[CmdletBinding()]
param(
  [switch]$ResetDb,
  [switch]$OnlyBackend,
  [switch]$OnlyFrontend,
  [int]$FrontendPort = 8088,
  [int]$BackendPort = 8000,
  [string]$PythonExe = 'python'
)

$ErrorActionPreference = 'Stop'

function Write-Info($m){ Write-Host "[fresh] $m" -ForegroundColor Cyan }
function Write-Warn($m){ Write-Host "[fresh] $m" -ForegroundColor Yellow }
function Write-Err ($m){ Write-Host "[fresh] $m" -ForegroundColor Red }

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
# Ascend until we find a directory containing 'backend' and 'frontend'
$candidate = $scriptDir
for ($i=0; $i -lt 4; $i++) {
  if (Test-Path (Join-Path $candidate 'backend/requirements.txt')) { break }
  $parent = Split-Path -Parent $candidate
  if ($parent -eq $candidate) { break }
  $candidate = $parent
}
if (-not (Test-Path (Join-Path $candidate 'backend/requirements.txt'))) {
  Write-Err "Fant ikke backend/requirements.txt ved oppstiging fra $scriptDir"
  exit 1
}
$repoRoot = $candidate
Set-Location $repoRoot
Write-Info "Repo rot: $repoRoot"

# 1. Reset DB om ønsket
if($ResetDb){
  $dbPath = Join-Path $repoRoot 'backend/data/trades.db'
  if(Test-Path $dbPath){
    Write-Info "Sletter DB: $dbPath"
    Remove-Item $dbPath -Force
  } else {
    Write-Warn "Ingen DB å slette: $dbPath"
  }
}

# 2. Backend setup
if(-not $OnlyFrontend){
  Write-Info 'Sikrer Python venv (.venv)'
  if(-not (Test-Path '.venv')){
      Write-Info 'Oppretter virtualenv (.venv) i rot'
      & $PythonExe -m venv .venv
    } else {
      Write-Info 'Bruker eksisterende .venv'
    }
    . .\.venv\Scripts\Activate.ps1
  Write-Info 'Installerer backend requirements (kan ta litt tid)'
  try { pip install --upgrade pip | Out-Null } catch { Write-Warn "Kunne ikke oppgradere pip: $($_.Exception.Message)" }
  pip install -r backend/requirements.txt | Out-Null
  Write-Info 'Starter backend (uvicorn)'
  Start-Job -Name qt_backend -ScriptBlock {
    param($Port,$RepoRoot)
    Set-Location $RepoRoot
    . .\.venv\Scripts\Activate.ps1
    uvicorn backend.main:app --host 127.0.0.1 --port $Port --log-level info
  } -ArgumentList $BackendPort,$repoRoot | Out-Null
  Start-Sleep -Seconds 2
  try {
    $ping = Invoke-WebRequest -UseBasicParsing ("http://127.0.0.1:"+$BackendPort+"/") -TimeoutSec 5
    Write-Info "Backend OK: status=$($ping.StatusCode)"
  } catch {
    Write-Err "Backend health feilet: $($_.Exception.Message)"
  }
}

# 3. Frontend prod build & static serve
if(-not $OnlyBackend){
  if(Test-Path 'frontend'){
    Push-Location frontend
    if(Test-Path 'dist'){ Remove-Item dist -Recurse -Force }
    Write-Info 'Installerer frontend avhengigheter (npm ci)'    
    if(Test-Path 'package-lock.json'){
      npm ci --no-audit --no-fund | Out-Null
    } else {
      npm install --no-audit --no-fund | Out-Null
    }
    Write-Info 'Bygger frontend (npm run build)'
    npm run build | Out-Null
    Pop-Location
    Write-Info "Starter statisk server på port $FrontendPort (python -m http.server)"
    Start-Job -Name qt_frontend -ScriptBlock {
      param($Port)
      Set-Location frontend/dist
      python -m http.server $Port
    } -ArgumentList $FrontendPort | Out-Null
    Start-Sleep -Seconds 2
    try {
      $html = Invoke-WebRequest -UseBasicParsing ("http://127.0.0.1:"+$FrontendPort+"/") -TimeoutSec 5
      Write-Info "Frontend OK: status=$($html.StatusCode) length=$($html.Content.Length)"
    } catch {
      Write-Err "Frontend fetch feilet: $($_.Exception.Message)"
    }
  } else {
    Write-Err 'Mangler frontend katalog.'
  }
}

Write-Info 'Ferdig. Aktive jobber:'
Get-Job | Where-Object { $_.Name -like 'qt_*' } | Format-Table Name,State,Id
