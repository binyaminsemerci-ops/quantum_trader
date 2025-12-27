try {
    $listenerPid = (Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty OwningProcess)
} catch {
    $listenerPid = $null
}

if ($listenerPid) {
    Write-Host "Stopping PID $listenerPid on :8000"
    try { Stop-Process -Id $listenerPid -Force -ErrorAction SilentlyContinue } catch {}
    Start-Sleep -Seconds 2
} else {
    Write-Host "No listener on :8000"
}
