# Converts remaining .jsx files under frontend/src -> .tsx
# If a target .tsx already exists, the original .jsx is moved to frontend/backups/unconverted/<original-path>
param(
    [string]$RepoRoot = "C:\quantum_trader"
)
Set-Location -Path $RepoRoot

function IsTracked($path) {
    try {
        git ls-files --error-unmatch $path *> $null
        return $true
    } catch {
        return $false
    }
}

$files = Get-ChildItem -Path "$RepoRoot\frontend\src" -Recurse -Filter *.jsx -File -ErrorAction SilentlyContinue
if (-not $files) { Write-Output "No .jsx files found under frontend/src"; exit 0 }

foreach ($f in $files) {
    $abs = $f.FullName
    # Relative path from repo root, Windows backslashes
    $rel = $abs.Substring($RepoRoot.Length + 1)
    $rel = $rel -replace '\\','\\'
    $targetRel = [System.IO.Path]::ChangeExtension($rel, '.tsx')
    $targetAbs = Join-Path $RepoRoot $targetRel

    Write-Output "Processing: $rel"

    if (Test-Path $targetAbs) {
        # Target exists — archive original .jsx to backups/unconverted/<original-path>
        $backupDir = Join-Path $RepoRoot (Join-Path "frontend\\backups\\unconverted" (Split-Path $rel -Parent))
        New-Item -ItemType Directory -Force -Path $backupDir | Out-Null
        $dest = Join-Path $backupDir (Split-Path $rel -Leaf)
        if (IsTracked $rel) {
            git mv -- "$rel" "$dest"
            Write-Output "Moved tracked $rel -> $dest (archived because target exists)"
        } else {
            Move-Item -LiteralPath $abs -Destination $dest -Force
            git add -- "$dest" 2>$null
            Write-Output "Moved untracked $rel -> $dest (archived because target exists)"
        }
    } else {
        # No target — convert (rename) to .tsx
        if (IsTracked $rel) {
            $ok = git mv -- "$rel" "$targetRel" 2>&1
            if ($?) { Write-Output "Renamed tracked $rel -> $targetRel" } else { Write-Output "git mv failed for $rel : $ok" }
        } else {
            New-Item -ItemType Directory -Force -Path (Split-Path $targetAbs -Parent) | Out-Null
            Move-Item -LiteralPath $abs -Destination $targetAbs -Force
            git add -- "$targetRel" 2>$null
            Write-Output "Renamed untracked $rel -> $targetRel and staged"
        }
    }
}

Write-Output 'Conversion script finished.'
