# Schedule weekly TFT performance review
# Run this script to set up automated weekly reviews

$scriptPath = "$PSScriptRoot\performance_review.py"
$taskName = "TFT_Weekly_Performance_Review"
$reviewDate = Get-Date "2025-11-26 09:00:00"

Write-Host "`n📅 SCHEDULING WEEKLY PERFORMANCE REVIEW" -ForegroundColor Cyan
Write-Host "="*60 -ForegroundColor Cyan

Write-Host "`n📋 Task Details:" -ForegroundColor White
Write-Host "   Name: $taskName"
Write-Host "   Script: $scriptPath"
Write-Host "   First Review: $($reviewDate.ToString('yyyy-MM-dd HH:mm'))"
Write-Host "   Frequency: Weekly (every Tuesday 9:00 AM)"

# Check if task exists
$existingTask = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue

if ($existingTask) {
    Write-Host "`n⚠️  Task already exists!" -ForegroundColor Yellow
    $response = Read-Host "Replace existing task? (y/N)"
    if ($response -ne 'y') {
        Write-Host "❌ Cancelled" -ForegroundColor Red
        exit
    }
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    Write-Host "✅ Removed existing task" -ForegroundColor Green
}

# Create scheduled task action
$action = New-ScheduledTaskAction `
    -Execute "python" `
    -Argument "$scriptPath" `
    -WorkingDirectory "C:\quantum_trader"

# Create trigger (weekly on Tuesday at 9 AM)
$trigger = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek Tuesday `
    -At $reviewDate

# Create principal (run as current user)
$principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive

# Create settings
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable

# Register task
try {
    Register-ScheduledTask `
        -TaskName $taskName `
        -Action $action `
        -Trigger $trigger `
        -Principal $principal `
        -Settings $settings `
        -Description "Weekly performance review for TFT trading model v1.1" | Out-Null
    
    Write-Host "`n✅ SCHEDULED TASK CREATED!" -ForegroundColor Green
    Write-Host "`n📊 You will receive automated performance reports every Tuesday at 9 AM" -ForegroundColor White
    Write-Host "`n🔍 To view task:" -ForegroundColor Yellow
    Write-Host "   Get-ScheduledTask -TaskName '$taskName'"
    Write-Host "`n🗑️  To remove task:" -ForegroundColor Yellow
    Write-Host "   Unregister-ScheduledTask -TaskName '$taskName' -Confirm:`$false"
    Write-Host "`n▶️  To run manually:" -ForegroundColor Yellow
    Write-Host "   python scripts\performance_review.py"
    
} catch {
    Write-Host "`n❌ FAILED TO CREATE TASK!" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
    Write-Host "`n💡 You may need administrator privileges" -ForegroundColor Yellow
    Write-Host "   Run PowerShell as Administrator and try again"
}

Write-Host "`n" + ("="*60) + "`n" -ForegroundColor Cyan
