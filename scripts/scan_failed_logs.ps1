$Dir = Join-Path (Get-Location) 'artifacts/ci-runs/gh-logs'
if (-not (Test-Path $Dir)) { Write-Host "No logs directory at $Dir"; exit 0 }
$files = Get-ChildItem -Path $Dir -Filter *.log -File | Sort-Object LastWriteTime -Descending
$patterns = 'Traceback','Exception','\bERROR\b','error:','failed','FAIL','AssertionError'
foreach ($f in $files) {
    $found = $false
    foreach ($p in $patterns) {
        $matches = Select-String -Path $f.FullName -Pattern $p -SimpleMatch:$false -ErrorAction SilentlyContinue
        if ($matches) {
            if (-not $found) { Write-Host "--- $($f.Name) ---"; $found = $true }
            $matches | Select-Object -First 6 | ForEach-Object { Write-Host "($($_.LineNumber)) $($_.Line.Trim())" }
        }
    }
    if ($found) { Write-Host "" }
}
Write-Host 'scan complete'
