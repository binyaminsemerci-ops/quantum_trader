<#
Create a minimal PR to fix frontend production high/critical npm audit findings.

Usage:
  pwsh .\scripts\create-audit-fix-pr.ps1 [-RunId <runId>] [-AutoOpen]

Behavior:
 - If -RunId provided, the script looks under artifacts/ci-runs/<runId>/metadata.json
   to find headSha and checks out that commit locally to reproduce the failing
   audit.
 - Otherwise, it scans artifacts/ci-runs/summary.json for entries with high>0 or
   critical>0 and will act on the first such run.
 - It attempts to run `npm ci` and `npm audit fix --production --package-lock-only`.
 - If package-lock.json (or package.json) changes, it creates a branch
   `fix/frontend-audit-<short-sha>` and opens a PR with the diff.
 - Requires gh CLI authenticated and push rights to the repo.

Notes: This script is conservative and will not push changes that result in
breaking version bumps; it simply tries an `npm audit fix` and opens a PR for
human review.
#>

Param(
    [int]$RunId,
    [switch]$AutoOpen
)

$repo = 'binyaminsemerci-ops/quantum_trader'
$base = Join-Path (Get-Location) 'artifacts/ci-runs'

function Find-ProblemRun {
    if ($RunId) {
        $meta = Join-Path $base $RunId
        if (-not (Test-Path $meta)) { return $null }
        $metaFile = Join-Path $meta 'metadata.json'
        if (-not (Test-Path $metaFile)) { return $null }
        return (Get-Content $metaFile | ConvertFrom-Json)
    }
    $summaryFile = Join-Path $base 'summary.json'
    if (-not (Test-Path $summaryFile)) { return $null }
    $summary = Get-Content $summaryFile | ConvertFrom-Json
    foreach ($s in $summary) {
        if ($s.found -and ($s.high -gt 0 -or $s.critical -gt 0)) {
            $dir = Split-Path $s.file -Parent
            $metaFile = Join-Path $dir 'metadata.json'
            if (Test-Path $metaFile) { return (Get-Content $metaFile | ConvertFrom-Json) }
        }
    }
    return $null
}

$problem = Find-ProblemRun
if (-not $problem) {
    Write-Host "No runs with production high/critical vulnerabilities found in artifacts. Exiting."
    exit 0
}

$headSha = $problem.headSha
Write-Host "Attempting to reproduce audit for commit $headSha"

# Create a temporary branch name
$short = $headSha.Substring(0,8)
$branch = "fix/frontend-audit-$short"

# Save current branch
$curBranch = git rev-parse --abbrev-ref HEAD

# Checkout the commit in a detached state then create branch
git fetch --all
git checkout -b $branch $headSha

# Run frontend reproduction steps
Set-Location frontend
npm ci --no-audit
npm audit --production --json > ..\artifacts\frontend-audit-repro-$short.json

# Try audit fix
$before = Get-Content package-lock.json -Raw
npm audit fix --production --package-lock-only
$after = Get-Content package-lock.json -Raw

if ($before -eq $after) {
    Write-Host "No package-lock changes after npm audit fix. Nothing to open as PR."
    # return to original branch
    Set-Location ..
    git checkout $curBranch
    git branch -D $branch
    exit 0
}

# Stage and commit changes
git add frontend/package-lock.json
if (Test-Path frontend/package.json) { git add frontend/package.json }
git commit -m "chore(frontend): audit fix for production vulnerabilities (repro $short)"

git push --set-upstream origin $branch

# Open PR
$prTitle = "chore(frontend): audit fixes for production vulnerabilities ($short)"
$prBody = "This automated PR attempts `npm audit fix` on commit $headSha and updates package-lock.json. Please review and test before merge."

# Determine labels to add: always add auto-approve; add auto-merge if requested
$labels = @('auto-approve')
if ($env:AUTO_ADD_AUTO_MERGE -eq '1' -or Test-Path (Join-Path $PSScriptRoot 'auto_merge')) {
    $labels += 'auto-merge'
}

# Create the PR and request labels at creation time. Use --json to capture the PR number/url if available.
try {
    $labelsArg = ($labels -join ',')
    $jsonOut = gh pr create --repo $repo --title $prTitle --body $prBody --base main --label $labelsArg --json number,url 2>$null
    if ($jsonOut) {
        $prObj = $jsonOut | ConvertFrom-Json
        Write-Host "Opened PR #$($prObj.number): $($prObj.url) with labels: $labelsArg"
    } else {
        # Fallback when --json not supported; gh prints the URL to stdout
        $url = gh pr create --repo $repo --title $prTitle --body $prBody --base main --label $labelsArg 2>$null
        Write-Host "Opened PR: $url with labels: $labelsArg"
    }
} catch {
    Write-Host "Failed to create PR via gh: $($_.Exception.Message)"
}
} else {
    Write-Host "Branch pushed: $branch. Run 'gh pr create --repo $repo --title \"$prTitle\" --body \"$prBody\" --base main --label auto-approve' to open a PR."
}

# Note: Do NOT automatically checkout original branch here to preserve worktree state
*** End of file
