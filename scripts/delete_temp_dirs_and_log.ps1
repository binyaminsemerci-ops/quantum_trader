$log = "C:\quantum_trader\scripts\delete_temp_dirs_and_log.txt"
"Log started: $(Get-Date)" | Out-File -FilePath $log -Encoding utf8
$dirs = @('C:\quantum_trader\qt_filterrepo_tmp','C:\quantum_trader\quantum_trader_filter_tmp','C:\quantum_trader\quantum_trader_filterrepo_mirror')
foreach ($d in $dirs) {
    "`n=== Processing: $d ===" | Out-File -FilePath $log -Append
    if (-not (Test-Path $d)) { "Not present: $d" | Out-File -FilePath $log -Append; continue }
    try {
        "Clearing readonly/hidden/system attributes (attrib -R -S -H)" | Out-File -FilePath $log -Append
        $attribOutput = cmd /c "attrib -R -S -H \"$d\\*.*\" /S /D" 2>&1
        $attribOutput | Out-File -FilePath $log -Append
    } catch { "attrib failed: $($_.Exception.Message)" | Out-File -FilePath $log -Append }
    try {
        "Attempting Remove-Item -Recurse -Force..." | Out-File -FilePath $log -Append
        Remove-Item -LiteralPath $d -Recurse -Force -ErrorAction Stop
        "Removed via Remove-Item: $d" | Out-File -FilePath $log -Append
        continue
    } catch {
        "Remove-Item failed: $($_.Exception.Message)" | Out-File -FilePath $log -Append
        "Remove-Item error details:" | Out-File -FilePath $log -Append
        $_ | Out-String | Out-File -FilePath $log -Append
    }
    try {
        "Attempting rmdir /S /Q..." | Out-File -FilePath $log -Append
        $rmdirOut = cmd /c "rmdir /S /Q \"$d\"" 2>&1
        $rmdirOut | Out-File -FilePath $log -Append
        if (-not (Test-Path $d)) { "Removed via rmdir: $d" | Out-File -FilePath $log -Append } else { "Still present after rmdir: $d" | Out-File -FilePath $log -Append }
    } catch {
        "rmdir failed: $($_.Exception.Message)" | Out-File -FilePath $log -Append
    }
}

"\nFinal presence check:" | Out-File -FilePath $log -Append
foreach ($d in $dirs) { "$d -> $(Test-Path $d)" | Out-File -FilePath $log -Append }

"Log ended: $(Get-Date)" | Out-File -FilePath $log -Append
