# Safe cleanup script: remove temporary debug scripts containing secrets and push a cleanup branch
param()

Write-Host "Starting cleanup script"

git fetch origin
git checkout -B fix/remove-secret-and-cleanup origin/feature/ci-frontend-build-and-tests
Write-Host "On branch:" (git rev-parse --abbrev-ref HEAD)

$files = @(
  'tmp_check_pr25.ps1',
  'query_runs_by_sha.ps1',
  'dispatch_and_monitor.ps1',
  'run_list_helper.ps1',
  'monitor_and_capture.ps1'
)

$staged = $false
foreach ($f in $files) {
  if (Test-Path $f) {
    Write-Host "Removing $f"
    git rm -f --ignore-unmatch $f
    $staged = $true
  } else {
    Write-Host "Not found: $f"
  }
}

if ($staged) {
  git commit -m 'chore(ci): remove tmp debug script (contained secret) and cleanup helper scripts'
  git push -u origin HEAD
  # Create PR from cleanup branch into feature branch
  gh pr create --title 'chore(ci): remove tmp debug script and cleanup helpers' --body 'Removes a temporary script containing a leaked token and deletes temporary helper scripts used for testing.' --base 'feature/ci-frontend-build-and-tests' --head 'fix/remove-secret-and-cleanup' --assume-yes
} else {
  Write-Host "No files staged for commit"
}

Write-Host "Cleanup script finished"
