if (-not (Test-Path 'backend/dev_in_runtime.txt')) {
  Write-Host "No backend/dev_in_runtime.txt found. Run the check first: python backend/scripts/check_dev_deps_in_runtime.py"
  exit 0
}

$pkgs = (Get-Content 'backend/dev_in_runtime.txt' -Raw).Split(',') | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne '' }
Write-Host "The following dev-only packages are detected in your runtime environment: $($pkgs -join ', ')"
$ans = Read-Host "Do you want to uninstall these packages now? [y/N]"
if ($ans -match '^[Yy]') {
  foreach ($p in $pkgs) {
    python -m pip uninstall -y $p
  }
  Write-Host "Uninstalled: $($pkgs -join ', ')"
} else {
  Write-Host "Aborted. No changes made."
}
