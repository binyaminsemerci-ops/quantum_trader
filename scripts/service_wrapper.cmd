@echo off
"C:\Program Files\PowerShell\7\pwsh.exe" -NoProfile -ExecutionPolicy Bypass -File "C:\quantum_trader\scripts\ci-watch.ps1" -PollIntervalSeconds 300
