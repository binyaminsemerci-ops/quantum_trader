# Creates a cleaned clone with tmp_check_pr25.ps1 removed and pushes a review branch
$repoUrl = 'https://github.com/binyaminsemerci-ops/quantum_trader.git'
$workDir = Join-Path $env:USERPROFILE 'quantum_trader_clean_tmp'
if (Test-Path $workDir) {
    Write-Output "Removing existing temp folder: $workDir"
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $workDir
}
Write-Output "Cloning $repoUrl -> $workDir"
git clone $repoUrl $workDir
if ($LASTEXITCODE -ne 0) { Write-Error 'git clone failed'; exit 1 }
Set-Location $workDir

Write-Output 'Installing git-filter-repo via pip in the active Python environment'
python -m pip install --upgrade pip
python -m pip install git-filter-repo
if ($LASTEXITCODE -ne 0) { Write-Error 'Failed to install git-filter-repo'; exit 1 }

Write-Output 'Running git-filter-repo to remove tmp_check_pr25.ps1 from history (local rewrite)'
git filter-repo --invert-paths --path tmp_check_pr25.ps1
if ($LASTEXITCODE -ne 0) { Write-Error 'git-filter-repo failed'; exit 1 }

# Create a review branch name
$branch = 'clean/remove-secret'
# Make sure we have a branch ref to base from; prefer origin/main then main
if (git show-ref --verify --quiet refs/remotes/origin/main) {
    git checkout -B $branch origin/main
} elseif (git show-ref --verify --quiet refs/heads/main) {
    git checkout -B $branch main
} else {
    git checkout -B $branch HEAD
}

Write-Output "Pushing cleaned branch $branch to origin (non-destructive)"
git push origin $branch
if ($LASTEXITCODE -ne 0) { Write-Error 'git push failed'; exit 1 }

Write-Output 'Cleaned branch pushed: clean/remove-secret (review before any force-pushes to existing branches)'
Write-Output "Temp workdir: $workDir"
Write-Output 'Done.'
