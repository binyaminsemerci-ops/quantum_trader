# Prints Python executable and version, checks if uvicorn is importable, starts uvicorn with logs redirected,
# waits briefly, reports whether port 8000 is listening, and prints the first 200 lines of the uvicorn log.

Write-Output "=== ENV INFO ==="
python -c "import sys; print('PYTHON_EXE=' + sys.executable); print('PYTHON_VER=' + sys.version.replace('\n',' '))"
python -c "import importlib.util; print('UVICORN_INSTALLED=' + str(importlib.util.find_spec('uvicorn') is not None))"

Write-Output "=== STARTING UVICORN ==="
# Ensure logs dir exists
if (-not (Test-Path -Path "logs")) { New-Item -ItemType Directory -Path "logs" | Out-Null }

# Build argument list for python
$argList = @('-u','-m','uvicorn','backend.main:app','--host','127.0.0.1','--port','8000','--log-level','info')

$proc = Start-Process -FilePath python -ArgumentList $argList -NoNewWindow -PassThru -RedirectStandardOutput "logs\\uvicorn.log" -RedirectStandardError "logs\\uvicorn.err.log"
Write-Output "Started uvicorn PID=$($proc.Id) at $([DateTime]::Now)"

Start-Sleep -Seconds 1

Write-Output "=== PORT LISTENING ==="
try {
    $listener = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction Stop | Select-Object LocalAddress,LocalPort,State,OwningProcess
    if ($listener) { $listener }
} catch {
    Write-Output "No listening socket found on port 8000"
}

Write-Output "=== UVICORN LOG (first 200 lines if present) ==="
if (Test-Path "logs\\uvicorn.log") {
    Get-Content "logs\\uvicorn.log" -TotalCount 200
} else {
    Write-Output "NO_LOG_FILE_YET"
}

Write-Output "=== STDERR (first 200 lines if present) ==="
if (Test-Path "logs\\uvicorn.err.log") {
    Get-Content "logs\\uvicorn.err.log" -TotalCount 200
} else {
    Write-Output "NO_ERR_LOG_FILE_YET"
}

Write-Output "=== DONE ==="
