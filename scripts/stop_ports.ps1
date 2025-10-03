$ports = @(8000, 5173)
foreach ($port in $ports) {
    try {
        $conns = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
        if ($conns) {
            $pids = $conns | Select-Object -ExpandProperty OwningProcess -Unique
            foreach ($pidVal in $pids) {
                try {
                    Stop-Process -Id $pidVal -Force -ErrorAction SilentlyContinue
                    Write-Output ("Stopped PID {0} on port {1}" -f $pidVal, $port)
                } catch {
                    Write-Output ("Failed to stop PID {0}: {1}" -f $pidVal, $_)
                }
            }
        } else {
            Write-Output ("No process on port {0}" -f $port)
        }
    } catch {
        Write-Output ("Error checking port {0}: {1}" -f $port, $_)
    }
}
