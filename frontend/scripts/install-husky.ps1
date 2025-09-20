<#
install-husky.ps1

Helper script to install frontend dependencies and run Husky prepare from PowerShell.
It attempts to find the repository root by walking up directories until it finds a `.git` folder.
If `.git` can't be found (CI/sandbox), it will still run `npm install --ignore-scripts` and print instructions.

Usage: run this from the repository root or from inside `frontend`:
  PS> .\frontend\scripts\install-husky.ps1
#>

function Find-GitRoot {
    param(
        [string]$StartDir
    )
    $cwd = if ($StartDir) { Get-Item $StartDir } else { Get-Location }
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
    # Allow override via environment variables for CI or non-standard layouts
    $envGitDir = $env:GIT_DIR
    $envRepoRoot = $env:REPO_ROOT

    # If GIT_DIR is provided but invalid, temporarily unset it so child processes
    # (npm/prepare/husky) don't inherit a broken path and fail with confusing errors.
    $oldGIT = $env:GIT_DIR
    $clearedGIT = $false
    if ($envGitDir) {
        Write-Host "GIT_DIR env provided: $envGitDir"
        if (Test-Path $envGitDir) {
            Write-Host "Using GIT_DIR to run prepare..."
            npm install
            npm run prepare
            Write-Host "Done. Husky hooks installed using GIT_DIR."
            return
        } else {
            Write-Warning "GIT_DIR provided but path not found: $envGitDir"
            try {
                Remove-Item Env:\GIT_DIR -ErrorAction SilentlyContinue
                $clearedGIT = $true
                Write-Host "Temporarily cleared invalid GIT_DIR for this session."
            } catch {
                Write-Warning "Could not clear GIT_DIR from environment: $_"
            }
        }
    }

    if ($envRepoRoot) {
        Write-Host "REPO_ROOT env provided: $envRepoRoot"
        $gitCandidate = Join-Path $envRepoRoot '.git'
        if (Test-Path $gitCandidate) {
            Write-Host "Found .git at $gitCandidate. Running normal install..."
            npm install
            npm run prepare
            Write-Host "Done. Husky hooks installed."
            return
        } else {
            Write-Warning "REPO_ROOT provided but .git not found at: $gitCandidate"
        }
    }

    $gitRoot = Find-GitRoot -StartDir (Get-Location)
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
        Write-Host "  # from PowerShell"
        Write-Host "  $env:REPO_ROOT = '<path-to-repo>'"
        Write-Host "  Set-Location (Join-Path $env:REPO_ROOT 'frontend')"
        Write-Host "  npm install"
        Write-Host "  npm run prepare"
        Write-Host "If your GitDir is custom, set GIT_DIR instead:"
        Write-Host "  $env:GIT_DIR = '<path-to-repo>\\.git'"
        Write-Host "  npm run prepare"
    }

    # restore GIT_DIR if we temporarily cleared it
    if ($clearedGIT -and $oldGIT) {
        try {
            $env:GIT_DIR = $oldGIT
            Write-Host "Restored original GIT_DIR environment variable."
        } catch {
            Write-Warning "Failed to restore original GIT_DIR: $_"
        }
    }
} finally {
    Pop-Location
}
