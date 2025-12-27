param(
  [string]$sha
)

if (-not $sha) {
  Write-Host "Usage: .\download_and_archive_runs.ps1 -sha <commit-sha>"
  exit 1
}

Write-Host "Finding workflow runs for SHA: $sha"
$runs = gh run list --repo binyaminsemerci-ops/quantum_trader --limit 200 --json databaseId,headSha,workflowName | ConvertFrom-Json
$matched = $runs | Where-Object { $_.headSha -eq $sha }
if (-not $matched) {
  Write-Host "No runs found for SHA $sha"
  exit 0
}

New-Item -ItemType Directory -Force -Path "artifacts\archived-runs" | Out-Null

foreach ($r in $matched) {
  $id = $r.databaseId
  $wf = $r.workflowName
  Write-Host "Downloading run $id ($wf) ..."
  try {
    gh run download $id --repo binyaminsemerci-ops/quantum_trader --dir "artifacts/archived-runs/$id"
  } catch {
    Write-Host ('Failed to download run ' + $id + ': ' + $_)
  }
}

Write-Host "Compressing artifacts/archived-runs -> artifacts/archived-runs.zip"
if (Test-Path 'artifacts/archived-runs.zip') { Remove-Item 'artifacts/archived-runs.zip' -Force }
Compress-Archive -Path 'artifacts\archived-runs\*' -DestinationPath 'artifacts/archived-runs.zip' -Force
Write-Host 'Created artifacts/archived-runs.zip'
