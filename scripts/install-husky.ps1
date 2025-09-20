<#
install-husky.ps1

Helper script to install frontend dependencies and run Husky prepare from PowerShell.
It attempts to find the repository root by walking up directories until it finds a `.git` folder.
If `.git` can't be found (CI/sandbox), it will still run `npm install --ignore-scripts` and print instructions.

Usage: run this from the repository root or from inside `frontend`:
  PS> .\frontend\scripts\install-husky.ps1
#>

function Find-GitRoot {
    $cwd = Get-Location
    while ($cwd -ne $null) {
        if (Test-Path (Join-Path $cwd '.git')) { return $cwd }
        $parent = Split-Path $cwd -Parent
        if ($parent -eq $cwd) { break }
        $cwd = Get-Item $parent
    }
    return $null
}

Write-Host "Installing frontend dependencies and preparing Husky hooks..."

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$frontendDir = Join-Path $scriptDir '..' | Resolve-Path -ErrorAction SilentlyContinue
if (-not $frontendDir) { $frontendDir = Join-Path (Get-Location) 'frontend' }
$frontendDir = (Get-Item $frontendDir).FullName

Write-Host "Frontend directory: $frontendDir"

Push-Location $frontendDir
try {
    $gitRoot = Find-GitRoot
    if ($gitRoot) {
        Write-Host "Found .git at $gitRoot. Running normal install..."
        npm install
        Write-Host "Running 'npm run prepare' to install Husky hooks..."
        npm run prepare
        Write-Host "Done. Husky hooks installed."
    } else {
        Write-Warning ".git not found. Installing dependencies without running lifecycle scripts (safe for CI/sandboxes)."
        npm install --ignore-scripts
        Write-Host "When on a developer machine with the repo available, run the following from the frontend folder to install Husky hooks:"
        Write-Host "  $env:GIT_DIR = '<path-to-repo>\\.git'"
        Write-Host "  npm run prepare"
    }
} finally {
    Pop-Location
}
