CI watcher README

This folder contains helper scripts to run and manage the CI watcher that polls
GitHub Actions for failed runs and downloads frontend audit artifacts.

Files:
- ci-watch.ps1 - Main watcher script that polls GH runs and downloads audit artifacts.
- register-ci-watch-task.ps1 - Register a Windows Scheduled Task named
  `QuantumTrader-CI-Watcher` to run the watcher at user logon.

How to use:
- To run the watcher now in background (already done by the agent):

  pwsh -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -FilePath pwsh -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','C:\\quantum_trader\\scripts\\ci-watch.ps1','-PollIntervalSeconds','300' -WorkingDirectory 'C:\\quantum_trader' -NoNewWindow -PassThru"

- To register the watcher as a Scheduled Task (run once as the user):

  pwsh .\\scripts\\register-ci-watch-task.ps1

Notes:
- The register script will create the scheduled task for the current user and
  run the watcher at logon. Adjust the task's trigger or arguments in the script
  if you want a different schedule or polling frequency.
