# Installs frontend deps and starts Vite (npm run dev) with logs redirected.
# Usage: run from repo root in PowerShell

$root = (Get-Location).Path
Write-Output "Working directory: $root"

# Stop existing node/npm
Stop-Process -Name node -Force -ErrorAction SilentlyContinue
Stop-Process -Name npm -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 1

# Ensure logs directory
$logsDir = Join-Path $root 'logs'
if (-not (Test-Path -Path $logsDir)) { New-Item -ItemType Directory -Path $logsDir | Out-Null }

# Run npm ci in frontend
$frontendDir = Join-Path $root 'frontend'
Write-Output "Installing frontend dependencies in $frontendDir (npm ci)"
Set-Location -Path $frontendDir
# Use npm.cmd for Windows
$npmCmd = 'npm.cmd'
$procCi = Start-Process -FilePath $npmCmd -ArgumentList 'ci','--no-audit','--no-fund' -NoNewWindow -PassThru -Wait
Write-Output "npm ci exit code: $($procCi.ExitCode)"

# Start Vite
$out = Join-Path $logsDir 'vite.log'
$err = Join-Path $logsDir 'vite.err.log'
Write-Output "Starting Vite via npm.cmd run dev (stdout -> $out, stderr -> $err)"
$proc = Start-Process -FilePath $npmCmd -ArgumentList 'run','dev' -WorkingDirectory $frontendDir -NoNewWindow -PassThru -RedirectStandardOutput $out -RedirectStandardError $err
Write-Output "Started npm.cmd PID=$($proc.Id)"
Start-Sleep -Seconds 6

Write-Output '=== LISTENER ON 5173 ==='
try { Get-NetTCPConnection -LocalPort 5173 -State Listen -ErrorAction Stop | Select-Object LocalAddress,LocalPort,State,OwningProcess } catch { Write-Output 'No listening socket on port 5173' }

Write-Output '=== VITE STDERR (tail 200) ==='
if (Test-Path $err) { Get-Content $err -Tail 200 } else { Write-Output 'vite.err.log missing' }
Write-Output '=== VITE STDOUT (tail 200) ==='
if (Test-Path $out) { Get-Content $out -Tail 200 } else { Write-Output 'vite.log missing' }

Write-Output '=== TEST PROXY /api/watchlist/prices ==='
try { $r = Invoke-RestMethod 'http://127.0.0.1:5173/api/watchlist/prices?symbols=BTCUSDT,ETHUSDT&limit=4' -ErrorAction Stop; $r } catch { Write-Output "PROXY REQUEST FAILED: $_" }

# Installs frontend deps and starts Vite (npm run dev) with logs redirected.
# Usage: run from repo root in PowerShell

$root = (Get-Location).Path
Write-Output "Working directory: $root"

# Stop existing node/npm
Stop-Process -Name node -Force -ErrorAction SilentlyContinue
Stop-Process -Name npm -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 1

# Ensure logs directory
$logsDir = Join-Path $root 'logs'
if (-not (Test-Path -Path $logsDir)) { New-Item -ItemType Directory -Path $logsDir | Out-Null }

# Run npm ci in frontend
$frontendDir = Join-Path $root 'frontend'
Write-Output "Installing frontend dependencies in $frontendDir (npm ci)"
Set-Location -Path $frontendDir
# Use npm.cmd for Windows
$npmCmd = 'npm.cmd'
$procCi = Start-Process -FilePath $npmCmd -ArgumentList 'ci','--no-audit','--no-fund' -NoNewWindow -PassThru -Wait
Write-Output "npm ci exit code: $($procCi.ExitCode)"

# Start Vite
$out = Join-Path $logsDir 'vite.log'
$err = Join-Path $logsDir 'vite.err.log'
Write-Output "Starting Vite via npm.cmd run dev (stdout -> $out, stderr -> $err)"
$proc = Start-Process -FilePath $npmCmd -ArgumentList 'run','dev' -WorkingDirectory $frontendDir -NoNewWindow -PassThru -RedirectStandardOutput $out -RedirectStandardError $err
Write-Output "Started npm.cmd PID=$($proc.Id)"
Start-Sleep -Seconds 6

Write-Output '=== LISTENER ON 5173 ==='
try { Get-NetTCPConnection -LocalPort 5173 -State Listen -ErrorAction Stop | Select-Object LocalAddress,LocalPort,State,OwningProcess } catch { Write-Output 'No listening socket on port 5173' }

Write-Output '=== VITE STDERR (tail 200) ==='
if (Test-Path $err) { Get-Content $err -Tail 200 } else { Write-Output 'vite.err.log missing' }
Write-Output '=== VITE STDOUT (tail 200) ==='
if (Test-Path $out) { Get-Content $out -Tail 200 } else { Write-Output 'vite.log missing' }

Write-Output '=== TEST PROXY /api/watchlist/prices ==='
try { $r = Invoke-RestMethod 'http://127.0.0.1:5173/api/watchlist/prices?symbols=BTCUSDT,ETHUSDT&limit=4' -ErrorAction Stop; $r } catch { Write-Output "PROXY REQUEST FAILED: $_" }

# Return to root
Set-Location -Path $root
Write-Output 'DONE'
