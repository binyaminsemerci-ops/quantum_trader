param(
  [switch]$EnableHooks
)

Write-Host "Bootstrapping development environment..."
. .\scripts\setup-dev.ps1  # Call the setup script (dot-source to run in the current session)

.
\scripts\setup-dev.ps1

if ($EnableHooks) {
  git config core.hooksPath .githooks
  Write-Host "Git hooks enabled."
} else {
  $ans = Read-Host "Enable local git hooks (configure core.hooksPath to .githooks)? [y/N]"
  if ($ans -match '^[Yy]') {
    git config core.hooksPath .githooks
    Write-Host "Git hooks enabled."
  } else {
    Write-Host "Skipping git hooks configuration."
  }
}

Write-Host "Bootstrap complete. Activate the venv with: .\\.venv\\Scripts\\Activate.ps1"
