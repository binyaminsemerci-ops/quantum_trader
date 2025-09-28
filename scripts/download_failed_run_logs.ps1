param(
    [string]$Repo = 'binyaminsemerci-ops/quantum_trader',
    [string]$Branch = 'main',
    [int]$Limit = 200
)

$OutDir = Join-Path $PSScriptRoot '..\artifacts\ci-runs\gh-logs' | Resolve-Path -Relative
$FullOutDir = Join-Path (Get-Location) $OutDir
if (-not (Test-Path $FullOutDir)) { New-Item -ItemType Directory -Path $FullOutDir -Force | Out-Null }

Write-Host "Fetching up to $Limit recent runs on branch '$Branch' for repo '$Repo'..."
$raw = gh run list --repo $Repo --branch $Branch --limit $Limit --json databaseId,name,headBranch,status,conclusion,createdAt 2>$null
if (-not $raw) {
    Write-Error "gh returned no output; ensure gh is authenticated and installed"
    exit 2
}

$runs = $raw | ConvertFrom-Json

$failed = @()
foreach ($r in $runs) {
    if ($r.conclusion -ne 'success') { $failed += $r }
}

if ($failed.Count -eq 0) { Write-Host "No non-success runs found."; exit 0 }

foreach ($r in $failed) {
    $id = $r.databaseId
    $safeName = ($r.name -replace '[^A-Za-z0-9_.-]', '_')
    $ts = Get-Date -Format 'yyyy_MM_dd_HH_mm_ss'
    $outFile = Join-Path $FullOutDir "run_${id}_${safeName}_${ts}.log"
    Write-Host "Fetching run $id ($($r.name)) -> $outFile"
    # gh expects the run-id as a positional arg, not --run-id
    gh run view $id --repo $Repo --log > $outFile
}

Write-Host "Saved logs to $FullOutDir"
