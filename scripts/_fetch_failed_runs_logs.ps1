# Fetch failed workflow runs and save failed-step logs for inspection
Param(
    [int]$Limit = 100
)
$Repo = 'binyaminsemerci-ops/quantum_trader'
$OutDir = Join-Path (Get-Location) 'artifacts/ci-runs/gh-logs'
if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir -Force | Out-Null }

Write-Host "Listing latest $Limit runs for repo $Repo"
$runsJson = gh run list --repo $Repo --limit $Limit --json databaseId,name,createdAt,conclusion 2>$null
if (-not $runsJson) { Write-Host 'gh run list returned nothing'; exit 0 }
$runs = $runsJson | ConvertFrom-Json
$failed = $runs | Where-Object { $_.conclusion -ne 'success' }
if (-not $failed -or $failed.Count -eq 0) { Write-Host 'No failed runs found'; exit 0 }

foreach ($r in $failed) {
    $id = $r.databaseId
    $name = $r.name -replace '[\\/:]', '-'    # sanitize
    $created = $r.createdAt -replace '[\\/: ]','_'
    $outfile = Join-Path $OutDir ("run_${id}_${name}_${created}.log")
    Write-Host "Fetching failed-step log for run $id ($name) -> $outfile"
    try {
        gh run view --repo $Repo $id --log-failed 2>&1 | Tee-Object -FilePath $outfile
    } catch {
        $err = $_.Exception.Message
        Write-Host "Failed to fetch logs for run ${id}: ${err}"
    }
    # also save job list JSON for reference
    try {
        $jobsJson = gh run view --repo $Repo $id --json jobs 2>$null
        if ($jobsJson) { $jobsJson | Set-Content -Path (Join-Path $OutDir ("run_${id}_jobs.json")) }
    } catch {}
}
Write-Host 'All logs fetched.'
