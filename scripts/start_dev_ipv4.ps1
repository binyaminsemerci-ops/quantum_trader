# Helper to stop project-related node/npm/uvicorn processes, start uvicorn and Vite bound to IPv4, and test endpoints
Param()

Set-StrictMode -Version Latest
# Compute repo root (parent of the scripts directory)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RepoRoot = Split-Path -Parent $scriptDir
Write-Output "Repository root: $RepoRoot"

Write-Output '=== STOP PROJECT PROCESSES (node/npm/uvicorn related to repo) ==='
# Find processes whose command line mentions our repo or frontend/backend folders
$procsFound = Get-CimInstance Win32_Process | Where-Object {
    ($_.CommandLine -and ($_.CommandLine -match 'quantum_trader|\\frontend|\\backend')) -and
    ($_.Name -match 'node|npm|node.exe|npm.cmd|python|python.exe|uvicorn')
}
if ($procsFound) {
    foreach ($p in $procsFound) {
        try {
            Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop
            Write-Output ("Stopped PID {0} ({1})" -f $p.ProcessId, $p.Name)
        } catch {
            Write-Output ("Failed to stop PID {0}: {1}" -f $p.ProcessId, $_.Exception.Message)
        }
    }
} else {
    Write-Output 'No matching project processes found.'
}

# Ensure logs directory exists
$logsDir = Join-Path $RepoRoot 'logs'
if (-not (Test-Path $logsDir)) { New-Item -ItemType Directory -Path $logsDir | Out-Null }

Write-Output '=== START UVICORN (127.0.0.1:8000) ==='
$uvOut = Join-Path $logsDir 'uvicorn.log'
$uvErr = Join-Path $logsDir 'uvicorn.err.log'

try {
    # Start uvicorn from repo root so Python can import the backend package
    $uvProc = Start-Process -FilePath 'python' -ArgumentList '-u','-m','uvicorn','backend.main:app','--host','127.0.0.1','--port','8000','--log-level','info' -NoNewWindow -PassThru -RedirectStandardOutput $uvOut -RedirectStandardError $uvErr -WorkingDirectory $RepoRoot
    Write-Output ("Started uvicorn PID={0}" -f $uvProc.Id)
} catch {
    Write-Output ("Failed to start uvicorn: {0}" -f $_.Exception.Message)
}

Start-Sleep -Seconds 2
Write-Output '=== LISTENER CHECK (8000) ==='
try { Get-NetTCPConnection -LocalPort 8000 -State Listen | Select-Object LocalAddress,LocalPort,State,OwningProcess } catch { Write-Output 'No listener on 8000' }

Write-Output '=== START FRONTEND (Vite) with IPv4 binding ==='

Set-Location -Path (Join-Path $RepoRoot 'frontend')

if (-not (Test-Path 'node_modules')) {
    if (Test-Path 'package-lock.json') {
        Write-Output 'node_modules missing and package-lock.json found — running npm.cmd ci'
        npm.cmd ci --no-audit --no-fund
    } else {
        Write-Output 'node_modules missing and no package-lock.json — running npm.cmd install'
        npm.cmd install --no-audit --no-fund
    }
} else {
    Write-Output 'node_modules present — skipping npm install/ci'
}

$viteOut = Join-Path $logsDir 'vite_ipv4.log'
$viteErr = Join-Path $logsDir 'vite_ipv4.err.log'

Write-Output ("Starting Vite (IPv4) via npm.cmd run dev -- --host 127.0.0.1 (logs -> $viteOut / $viteErr)")
try {
    $viteProc = Start-Process -FilePath 'npm.cmd' -ArgumentList 'run','dev','--','--host','127.0.0.1' -WorkingDirectory (Join-Path $RepoRoot 'frontend') -NoNewWindow -PassThru -RedirectStandardOutput $viteOut -RedirectStandardError $viteErr
    if ($viteProc -and $viteProc.Id) { Write-Output ("Started npm.cmd PID={0}" -f $viteProc.Id) } else { Write-Output 'Started npm.cmd (no PID available)' }
} catch {
    Write-Output ("Failed to start Vite: {0}" -f $_.Exception.Message)
}

Start-Sleep -Seconds 5
Write-Output '=== LISTENER CHECK (5173) ==='
try { Get-NetTCPConnection -LocalPort 5173 -State Listen | Select-Object LocalAddress,LocalPort,State,OwningProcess } catch { Write-Output 'No listener on 5173' }

Write-Output '=== TEST BACKEND DIRECT ==='
try {
    $backendRes = Invoke-RestMethod 'http://127.0.0.1:8000/watchlist/prices?symbols=BTCUSDT,ETHUSDT&limit=2' -ErrorAction Stop
    Write-Output 'BACKEND OK — sample response (trimmed)'
    $backendRes | ConvertTo-Json -Depth 2 | Select-Object -First 1
} catch {
    Write-Output ("BACKEND TEST FAILED: {0}" -f $_.Exception.Message)
}

Write-Output '=== TEST FRONTEND PROXY ==='
try {
    $proxyRes = Invoke-RestMethod 'http://127.0.0.1:5173/api/watchlist/prices?symbols=BTCUSDT,ETHUSDT&limit=2' -ErrorAction Stop
    Write-Output 'PROXY OK — sample response (trimmed)'
    $proxyRes | ConvertTo-Json -Depth 2 | Select-Object -First 1
} catch {
    Write-Output ("PROXY TEST FAILED: {0}" -f $_.Exception.Message)
}

Write-Output '=== TAIL LOGS (short excerpts) ==='
if (Test-Path $uvErr) { Write-Output '--- uvicorn.err.log ---'; Get-Content $uvErr -Tail 200 } else { Write-Output 'uvicorn.err.log missing' }
if (Test-Path $uvOut) { Write-Output '--- uvicorn.log ---'; Get-Content $uvOut -Tail 200 } else { Write-Output 'uvicorn.log missing' }
if (Test-Path $viteErr) { Write-Output '--- vite_ipv4.err.log ---'; Get-Content $viteErr -Tail 200 } else { Write-Output 'vite_ipv4.err.log missing' }
if (Test-Path $viteOut) { Write-Output '--- vite_ipv4.log ---'; Get-Content $viteOut -Tail 200 } else { Write-Output 'vite_ipv4.log missing' }

Write-Output '=== DONE ==='
