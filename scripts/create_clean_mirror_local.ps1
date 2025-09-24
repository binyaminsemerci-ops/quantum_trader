# Safer local mirror-based cleaning (works inside workspace)
$repoUrl = 'https://github.com/binyaminsemerci-ops/quantum_trader.git'
$mirrorDir = Join-Path (Get-Location) 'quantum_trader_mirror.git'
if (Test-Path $mirrorDir) { Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $mirrorDir }
Write-Output "Cloning mirror to $mirrorDir"
git clone --mirror $repoUrl $mirrorDir
if ($LASTEXITCODE -ne 0) { Write-Error 'mirror clone failed'; exit 1 }

Set-Location $mirrorDir

# Ensure git-filter-repo is available
Write-Output 'Installing git-filter-repo into active Python environment (if needed)'
python -m pip install --upgrade pip
python -m pip install git-filter-repo

Write-Output 'Running git-filter-repo to remove tmp_check_pr25.ps1 from mirror'
try {
    git filter-repo --invert-paths --path tmp_check_pr25.ps1
} catch {
    Write-Error 'git-filter-repo failed: ' $_.Exception.Message
    exit 1
}

# Create a temporary branch for review from mirror's refs/heads/main (if present)
if (Test-Path .\refs\heads\main) {
    git symbolic-ref HEAD refs/heads/main
    git branch -f clean/remove-secret main
} elseif (git for-each-ref --format='%(refname)' refs/heads | Select-String -Pattern 'main') {
    git branch -f clean/remove-secret refs/heads/main
} else {
    # pick any head
    $head = git for-each-ref --format='%(refname:strip=2)' refs/heads | Select-Object -First 1
    git branch -f clean/remove-secret $head
}

# Push cleaned branch back to origin (non-destructive)
Write-Output 'Pushing clean/remove-secret to origin (non-destructive)'
git push origin refs/heads/clean/remove-secret:refs/heads/clean/remove-secret
if ($LASTEXITCODE -ne 0) { Write-Error 'git push failed'; exit 1 }

Write-Output 'Clean branch pushed: clean/remove-secret'
Write-Output "Mirror dir: $mirrorDir"
