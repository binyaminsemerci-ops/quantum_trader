# STRESS TEST MONITORING SCRIPT
# Collects data over 10 minutes to verify system stability

$logFile = "C:\quantum_trader\stress_test_log.txt"
$startTime = Get-Date
$endTime = $startTime.AddMinutes(10)
$cycleCount = 0
$errorCount = 0
$warningCount = 0

Write-Output "=== STRESS TEST MONITORING ===" | Out-File $logFile
Write-Output "Start: $startTime" | Out-File $logFile -Append
Write-Output "End Target: $endTime" | Out-File $logFile -Append
Write-Output "" | Out-File $logFile -Append

# Baseline metrics
Write-Output "=== BASELINE METRICS ===" | Out-File $logFile -Append
$baseline = docker logs quantum_backend --since 30s 2>&1 | Select-String -Pattern "AI-HFOS|SELF-HEAL|PORTFOLIO|MODEL" | Measure-Object
Write-Output "Recent log entries: $($baseline.Count)" | Out-File $logFile -Append
Write-Output "" | Out-File $logFile -Append

# Monitor for 10 minutes
$checkInterval = 60 # Check every 60 seconds
$checksToRun = 10

for ($i = 1; $i -le $checksToRun; $i++) {
    $currentTime = Get-Date
    Write-Output "=== CHECK $i at $currentTime ===" | Out-File $logFile -Append
    
    # Check AI-HFOS coordination
    $hfos = docker logs quantum_backend --since 70s 2>&1 | Select-String -Pattern "AI-HFOS.*Coordination complete"
    $hfosCount = ($hfos | Measure-Object).Count
    Write-Output "  AI-HFOS cycles: $hfosCount" | Out-File $logFile -Append
    
    # Check Self-Healing
    $heal = docker logs quantum_backend --since 70s 2>&1 | Select-String -Pattern "SELF-HEAL.*Health check complete"
    $healCount = ($heal | Measure-Object).Count
    Write-Output "  Self-Healing checks: $healCount" | Out-File $logFile -Append
    
    # Check errors
    $errors = docker logs quantum_backend --since 70s 2>&1 | Select-String -Pattern "ERROR" | Where-Object { $_ -notmatch "SELF-HEAL.*CRITICAL ISSUES" }
    $errCount = ($errors | Measure-Object).Count
    Write-Output "  Unexpected errors: $errCount" | Out-File $logFile -Append
    $errorCount += $errCount
    
    # Check for task crashes
    $crashes = docker logs quantum_backend --since 70s 2>&1 | Select-String -Pattern "cancelled|CancelledError|task.*failed"
    $crashCount = ($crashes | Measure-Object).Count
    Write-Output "  Task crashes: $crashCount" | Out-File $logFile -Append
    
    # Check Universe OS
    $universe = docker logs quantum_backend --since 70s 2>&1 | Select-String -Pattern "Universe OS.*processing"
    $universeCount = ($universe | Measure-Object).Count
    Write-Output "  Universe OS cycles: $universeCount" | Out-File $logFile -Append
    
    # Check Dynamic TP/SL
    $tpsl = docker logs quantum_backend --since 70s 2>&1 | Select-String -Pattern "Dynamic TP/SL"
    $tpslCount = ($tpsl | Measure-Object).Count
    Write-Output "  Dynamic TP/SL calculations: $tpslCount" | Out-File $logFile -Append
    
    Write-Output "" | Out-File $logFile -Append
    
    if ($i -lt $checksToRun) {
        Start-Sleep -Seconds $checkInterval
    }
}

# Summary
$endTimeActual = Get-Date
Write-Output "=== STRESS TEST COMPLETE ===" | Out-File $logFile -Append
Write-Output "End: $endTimeActual" | Out-File $logFile -Append
Write-Output "Duration: $($endTimeActual - $startTime)" | Out-File $logFile -Append
Write-Output "Total unexpected errors: $errorCount" | Out-File $logFile -Append
Write-Output "" | Out-File $logFile -Append

Write-Host "Stress test monitoring complete. Results in: $logFile"
