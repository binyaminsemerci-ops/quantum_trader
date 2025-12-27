<#
Create a per-user Startup CMD file that starts the CI watcher at login.
Usage: pwsh -NoProfile -ExecutionPolicy Bypass -File .\scripts\create_startup.ps1
#>

$startupFolder = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs\Startup'
if (-not (Test-Path $startupFolder)) { New-Item -ItemType Directory -Path $startupFolder -Force | Out-Null }

$startupFile = Join-Path $startupFolder 'quantum_trader_ci_watch_startup.cmd'

$pwshPath = (Get-Command pwsh -ErrorAction SilentlyContinue).Source
if (-not $pwshPath) { $pwshPath = 'pwsh' }

$watcherPath = (Resolve-Path '.\scripts\ci-watch.ps1').ProviderPath

$line1 = '@echo off'
$line2 = 'start "" "' + $pwshPath + '" -NoProfile -ExecutionPolicy Bypass -File "' + $watcherPath + '" -PollIntervalSeconds 300'
$line3 = 'exit'

$content = $line1 + "`r`n" + $line2 + "`r`n" + $line3 + "`r`n"

Set-Content -Path $startupFile -Value $content -Encoding ASCII -Force
Write-Host "WROTE STARTUP FILE: $startupFile"
Write-Host '---- contents ----'
Get-Content $startupFile | ForEach-Object { Write-Host $_ }
Write-Host '---- end ----'

Write-Host "The watcher will start automatically when you next log in (per-user Startup)."
