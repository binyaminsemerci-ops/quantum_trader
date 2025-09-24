# Create a regular clone, run git filter-branch to remove tmp_check_pr25.ps1, and push a cleaned branch for review
$repoUrl = 'https://github.com/binyaminsemerci-ops/quantum_trader.git'
$workDir = Join-Path (Get-Location) 'quantum_trader_filter_tmp'
if (Test-Path $workDir) { Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $workDir }

Write-Output "Cloning $repoUrl -> $workDir"
git clone $repoUrl $workDir
if ($LASTEXITCODE -ne 0) { Write-Error 'git clone failed'; exit 1 }
Set-Location $workDir

Write-Output 'Running git filter-branch to remove tmp_check_pr25.ps1 from all refs (local rewrite)'
# Use index-filter for speed
$cmd = 'git filter-branch --force --index-filter "git rm --cached --ignore-unmatch tmp_check_pr25.ps1" --prune-empty --tag-name-filter cat -- --all'
Write-Output "Running: $cmd"
Invoke-Expression $cmd
if ($LASTEXITCODE -ne 0) { Write-Error 'git filter-branch failed'; exit 1 }

# Cleanup refs/original to reduce confusion (local only)
Remove-Item -Recurse -Force .git\refs\original -ErrorAction SilentlyContinue

# Run garbage collection
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# Create a cleaned branch from main (if present) or current HEAD
$branch = 'clean/remove-secret'
if (git show-ref --verify --quiet refs/heads/main) {
    git branch -f $branch main
} elseif (git show-ref --verify --quiet refs/remotes/origin/main) {
    git branch -f $branch refs/remotes/origin/main
} else {
    git branch -f $branch HEAD
}

Write-Output "Pushing cleaned branch $branch to origin (non-destructive)"
git push origin $branch
if ($LASTEXITCODE -ne 0) { Write-Error 'git push failed'; exit 1 }

Write-Output 'Cleaned branch pushed: clean/remove-secret'
Write-Output "Local workdir: $workDir"
