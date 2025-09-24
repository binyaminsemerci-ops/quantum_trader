$ErrorActionPreference='Stop'

Write-Host "Fetching origin..."
git fetch origin

$default = (git rev-parse --abbrev-ref origin/HEAD)
if ($default -match '/') { $default = $default.Split('/')[1] }
Write-Host "Default branch: $default"

$ts = Get-Date -Format 'yyyyMMddHHmmss'
$branch = "chore/add-auto-labeler-to-$($default)-$ts"
Write-Host "Creating branch $branch from origin/$default"

git checkout -b $branch origin/$default

# Ensure directory exists
New-Item -ItemType Directory -Force -Path .github\workflows | Out-Null

$srcBranch = 'feature/ci-frontend-build-and-tests'
$srcPath = '.github/workflows/auto_label_on_comment.yml'
$destPath = '.github/workflows/auto_label_on_comment.yml'

try {
    # Use single quotes and a local variable to avoid PowerShell interpreting the colon as a variable delimiter
    $gitShowArg = "$($srcBranch):$($srcPath)"
    git show $gitShowArg > $destPath
    Write-Host "Extracted $srcPath from $srcBranch"
} catch {
    Write-Error ("Failed to extract file from {0}: {1}" -f $srcBranch, $_)
    exit 1
}

# Stage and commit only if there are changes
git add $destPath
$staged = git diff --cached --name-only
if ($staged) {
    git commit -m 'chore: add auto-ruff-fix labeler workflow'
    Write-Host 'Committed changes'
} else {
    Write-Host 'No changes to commit (file may already be identical on target branch)'
}

Write-Host "Pushing branch $branch to origin..."
git push -u origin $branch

if (Get-Command gh -ErrorAction SilentlyContinue) {
    Write-Host 'gh CLI found — attempting to create PR...'
    $prJson = gh pr create --title 'chore: add auto-ruff-fix labeler workflow' --body 'Add policy and labeler workflow to allow trusted commenters to opt-in for ruff auto-commits via label auto-ruff-fix.' --base $default --head $branch --repo binyaminsemerci-ops/quantum_trader --json url
    $url = ($prJson | ConvertFrom-Json).url
    Write-Host "PR created: $url"
} else {
    Write-Host 'gh CLI not found; branch pushed. To create a PR run:'
    Write-Host "gh pr create --title 'chore: add auto-ruff-fix labeler workflow' --body 'Add policy and labeler workflow to allow trusted commenters to opt-in for ruff auto-commits via label auto-ruff-fix.' --base $default --head $branch --repo binyaminsemerci-ops/quantum_trader"
    Write-Host "Or open a PR at: https://github.com/binyaminsemerci-ops/quantum_trader/compare/$default...$branch"
}
