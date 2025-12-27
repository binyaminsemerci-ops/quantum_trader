<#!
.SYNOPSIS
  Cleanly restarts Quantum Trader backend (FastAPI) on port 8080 and frontend Vite dev server.

.DESCRIPTION
  Kills any processes listening on the target backend port (default 8080) and frontend port (default 5173),
  then starts the backend (without reload by default to avoid rapid reload loop) and the frontend dev server.

.PARAMETER BackendPort
  Backend port (default 8080)

.PARAMETER FrontendPort
  Frontend dev port (default 5173)

.PARAMETER Reload
  Set to $true to enable uvicorn reload (adds exclusions to reduce churn).

.EXAMPLE
  ./scripts/restart-dev.ps1

.EXAMPLE
  ./scripts/restart-dev.ps1 -BackendPort 8181 -Reload $true

#>
param(
  [int]$BackendPort = 8080,
  [int]$FrontendPort = 5173,
  [switch]$Reload
)

Write-Host "[restart-dev] BackendPort=$BackendPort FrontendPort=$FrontendPort Reload=$Reload"

function Stop-PortListeners {
  param([int[]]$Ports)
  $conns = Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object { $_.LocalPort -in $Ports }
  if(-not $conns){ Write-Host "[restart-dev] No listeners on ports: $Ports"; return }
  $pids = $conns | Select-Object -ExpandProperty OwningProcess -Unique
  foreach($pid in $pids){
    try {
      $proc = Get-Process -Id $pid -ErrorAction SilentlyContinue
      if($proc){
        Write-Host "[restart-dev] Stopping PID $pid ($($proc.ProcessName))"
        Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
      }
    } catch { }
  }
}

Push-Location (Split-Path $PSScriptRoot -Parent)

# 1. Kill old listeners
Stop-PortListeners -Ports @($BackendPort, $FrontendPort, 8000)

# 2. Start backend
Write-Host "[restart-dev] Starting backend..."
Push-Location backend

$env:PORT = "$BackendPort"
$env:UVICORN_RELOAD = ($Reload.IsPresent ? '1' : '0')

$backendCmd = if($Reload){
  "python -m uvicorn main:app --host 0.0.0.0 --port $BackendPort --reload --reload-exclude logs --reload-exclude *.db"
} else {
  "python -m uvicorn main:app --host 0.0.0.0 --port $BackendPort"
}

# Start in its own window for logs
Start-Process -FilePath powershell -ArgumentList "-NoLogo","-NoProfile","-Command","cd `"$PWD`"; $backendCmd" -WindowStyle Normal | Out-Null
Pop-Location

Start-Sleep -Seconds 2

# 3. Start frontend
Write-Host "[restart-dev] Starting frontend..."
Push-Location frontend

if(Test-Path package.json){
  # Ensure node_modules exists
  if(-not (Test-Path node_modules)){
    Write-Host "[restart-dev] Installing frontend dependencies (first run)..."
    npm install
  }
  $env:VITE_BACKEND_PORT = "$BackendPort"
  Start-Process -FilePath powershell -ArgumentList "-NoLogo","-NoProfile","-Command","cd `"$PWD`"; npm run dev" -WindowStyle Normal | Out-Null
} else {
  Write-Warning "[restart-dev] package.json not found in frontend directory."
}
Pop-Location

Write-Host "[restart-dev] Done. Backend: http://localhost:$BackendPort  Frontend: http://localhost:$FrontendPort"
Pop-Location