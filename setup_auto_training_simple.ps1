# setup_auto_training_simple.ps1 - Simple setup for automatic AI training

param([string]$Action = "help")

$TaskName = "Quantum Trader AI Auto Training"

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "🤖 QUANTUM TRADER AUTO TRAINING SETUP" -ForegroundColor Yellow  
Write-Host "================================================" -ForegroundColor Cyan

switch ($Action.ToLower()) {
    "install" {
        Write-Host "📝 Installing automatic training..." -ForegroundColor Green
        try {
            $xmlPath = "C:\quantum_trader\quantum_trader_auto_training.xml"
            if (Test-Path $xmlPath) {
                Register-ScheduledTask -TaskName $TaskName -Xml (Get-Content $xmlPath | Out-String) -Force
                Write-Host "✅ Auto training installed successfully!" -ForegroundColor Green
                Write-Host "🔄 Will run at startup and hourly" -ForegroundColor Yellow
            } else {
                Write-Host "❌ XML file not found: $xmlPath" -ForegroundColor Red
            }
        } catch {
            Write-Host "❌ Failed to install: $_" -ForegroundColor Red
        }
    }
    "remove" {
        Write-Host "🗑️ Removing automatic training..." -ForegroundColor Yellow
        try {
            Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction Stop
            Write-Host "✅ Auto training removed!" -ForegroundColor Green
        } catch {
            Write-Host "❌ Task not found or failed to remove" -ForegroundColor Red
        }
    }
    "status" {
        Write-Host "📊 Checking status..." -ForegroundColor Blue
        $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
        if ($task) {
            Write-Host "✅ Auto training is INSTALLED" -ForegroundColor Green
            Write-Host "📅 State: $($task.State)" -ForegroundColor White
        } else {
            Write-Host "❌ Auto training is NOT installed" -ForegroundColor Red
        }
    }
    "test" {
        Write-Host "🧪 Running test training..." -ForegroundColor Blue
        cd C:\quantum_trader
        & ".\start_training_optimized.bat" 20
    }
    default {
        Write-Host "🚀 USAGE:" -ForegroundColor Green
        Write-Host "  .\setup_auto_training_simple.ps1 install" -ForegroundColor White
        Write-Host "  .\setup_auto_training_simple.ps1 remove" -ForegroundColor White  
        Write-Host "  .\setup_auto_training_simple.ps1 status" -ForegroundColor White
        Write-Host "  .\setup_auto_training_simple.ps1 test" -ForegroundColor White
    }
}

Write-Host "================================================" -ForegroundColor Cyan