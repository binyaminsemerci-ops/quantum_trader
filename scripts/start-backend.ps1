param(
    [switch]$NewWindow,
    [int]$Port = 8000,
    [string]$BindHost = "0.0.0.0",
    [switch]$NoReload
)

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSCommandPath | Split-Path -Parent
$venvActivate = Join-Path $root ".venv/Scripts/Activate.ps1"
if (Test-Path $venvActivate) {
    . $venvActivate
}

$env:PYTHONUNBUFFERED = "1"

Write-Host "PowerShell $($PSVersionTable.PSVersion)" -ForegroundColor DarkGray

$interesting = @(
    'DB_URL','BINANCE_API_KEY','BINANCE_API_SECRET','DEFAULT_BALANCE','DEFAULT_RISK_PERCENT',
    'GHCR_NAMESPACE','GHCR_USERNAME','GHCR_PAT','QT_XGB_THRESHOLD','QT_ADMIN_TOKEN'
)
foreach ($name in $interesting) {
    if ((Get-Item "env:$name" -ErrorAction SilentlyContinue).Value) { 
        Write-Host "Loaded: $name" -ForegroundColor DarkGray 
    }
}

Write-Host "Starting backend on port $Port..." -ForegroundColor Yellow

$reloadArg = if ($NoReload) { @() } else { @('--reload') }
$pythonArgs = @('-m', 'uvicorn', 'backend.main:app', '--host', $BindHost, '--port', $Port) + $reloadArg

if ($NewWindow) {
    Start-Process -WorkingDirectory $root -FilePath "python" -ArgumentList $pythonArgs -WindowStyle Minimized | Out-Null
    Start-Sleep 2
    Write-Host "Launched Uvicorn in a new window. Checking health..." -ForegroundColor Cyan
    try {
        (Invoke-WebRequest -UseBasicParsing "http://localhost:$Port/health" -TimeoutSec 5).Content | Write-Output
    } catch {
        Write-Host $_.Exception.Message -ForegroundColor Red
    }
} else {
    python $pythonArgs
}
