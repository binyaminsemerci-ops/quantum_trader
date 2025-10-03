# Auto Training Setup for Quantum Trader AI
param([string]$Action = "help")

$TaskName = "Quantum Trader AI Auto Training"

Write-Host "Quantum Trader Auto Training Setup" -ForegroundColor Cyan

switch ($Action.ToLower()) {
    "install" {
        # Check if running as Administrator
        $isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
        
        if (-not $isAdmin) {
            Write-Host "ERROR: Administrator rights required!" -ForegroundColor Red
            Write-Host "Solution 1: Right-click PowerShell -> 'Run as Administrator'" -ForegroundColor Yellow
            Write-Host "Solution 2: Use startup folder method instead:" -ForegroundColor Yellow
            Write-Host "  .\setup_simple.ps1 startup" -ForegroundColor Cyan
            return
        }
        
        Write-Host "Installing automatic training..." -ForegroundColor Green
        try {
            $xmlPath = "C:\quantum_trader\quantum_trader_auto_training.xml"
            if (Test-Path $xmlPath) {
                Register-ScheduledTask -TaskName $TaskName -Xml (Get-Content $xmlPath | Out-String) -Force -ErrorAction Stop
                Write-Host "SUCCESS: Auto training installed!" -ForegroundColor Green
                Write-Host "Will run at startup and every hour" -ForegroundColor Yellow
            } else {
                Write-Host "ERROR: XML file not found: $xmlPath" -ForegroundColor Red
            }
        } catch {
            Write-Host "ERROR: Failed to install - $($_.Exception.Message)" -ForegroundColor Red
        }
    }
    "remove" {
        Write-Host "Removing automatic training..." -ForegroundColor Yellow
        try {
            Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction Stop
            Write-Host "SUCCESS: Auto training removed!" -ForegroundColor Green
        } catch {
            Write-Host "ERROR: Task not found or failed to remove" -ForegroundColor Red
        }
    }
    "status" {
        Write-Host "Checking status..." -ForegroundColor Blue
        $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
        if ($task) {
            Write-Host "SUCCESS: Auto training is INSTALLED" -ForegroundColor Green
            Write-Host "State: $($task.State)" -ForegroundColor White
        } else {
            Write-Host "INFO: Auto training is NOT installed" -ForegroundColor Yellow
        }
    }
    "startup" {
        Write-Host "Installing to startup folder (no admin required)..." -ForegroundColor Green
        try {
            $startupFolder = [Environment]::GetFolderPath("Startup")
            $shortcutPath = Join-Path $startupFolder "Quantum Trader AI.lnk"
            $batPath = "C:\quantum_trader\auto_training_scheduler.bat"
            
            # Create shortcut using COM object
            $WshShell = New-Object -comObject WScript.Shell
            $Shortcut = $WshShell.CreateShortcut($shortcutPath)
            $Shortcut.TargetPath = $batPath
            $Shortcut.WorkingDirectory = "C:\quantum_trader"
            $Shortcut.Description = "Quantum Trader AI Auto Training"
            $Shortcut.Save()
            
            Write-Host "SUCCESS: Added to startup folder!" -ForegroundColor Green
            Write-Host "Location: $shortcutPath" -ForegroundColor Yellow
            Write-Host "Will start training when you log in" -ForegroundColor Yellow
        } catch {
            Write-Host "ERROR: Failed to create startup shortcut - $($_.Exception.Message)" -ForegroundColor Red
        }
    }
    "test" {
        Write-Host "Running test training..." -ForegroundColor Blue
        Set-Location C:\quantum_trader
        & ".\start_training_optimized.bat" 20
    }
    default {
        Write-Host "USAGE:" -ForegroundColor Green
        Write-Host "  .\setup_simple.ps1 install  (requires admin)" -ForegroundColor White
        Write-Host "  .\setup_simple.ps1 startup  (no admin needed)" -ForegroundColor Cyan
        Write-Host "  .\setup_simple.ps1 remove" -ForegroundColor White  
        Write-Host "  .\setup_simple.ps1 status" -ForegroundColor White
        Write-Host "  .\setup_simple.ps1 test" -ForegroundColor White
        Write-Host ""
        Write-Host "RECOMMENDED: Use 'startup' method (easier)" -ForegroundColor Yellow
    }
}