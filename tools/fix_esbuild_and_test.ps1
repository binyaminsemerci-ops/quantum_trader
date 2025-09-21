<#
tools/fix_esbuild_and_test.ps1

Safe helper for Windows to: inspect for processes holding esbuild, prompt to kill them,
remove the esbuild binary if present, run `npm ci --ignore-scripts` in frontend, and run tests.

Usage (run from repo root):
  pwsh -NoProfile -File .\tools\fix_esbuild_and_test.ps1
  pwsh -NoProfile -File .\tools\fix_esbuild_and_test.ps1 -AutoKill

The script is conservative: it prints findings and prompts before killing processes. Use -AutoKill to skip prompts.
#>

param(
  [switch]$AutoKill = $false
)

function Find-LockingProcesses {
  Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'node' -or $_.Name -match 'esbuild' } | Select-Object ProcessId, Name, CommandLine
}

Write-Host "== fix_esbuild_and_test.ps1: Inspecting processes that may hold esbuild.exe =="
$procs = Find-LockingProcesses
if ($procs) {
  $procs | Format-Table -AutoSize
} else {
  Write-Host "No node/esbuild processes detected."
}

if (-not $AutoKill -and $procs) {
  $ans = Read-Host "Kill the listed processes? (y/N)"
  if ($ans -ne 'y' -and $ans -ne 'Y') {
    Write-Host "Aborting per user request. Close processes manually and re-run this script."
    exit 0
  }
}

if ($procs) {
  foreach ($p in $procs) {
    try {
      Write-Host "Stopping PID $($p.ProcessId) ($($p.Name))..."
      Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop
    } catch {
      Write-Warning "Failed to stop PID $($p.ProcessId): $_"
    }
  }
}

# Remove esbuild binary if present inside frontend node_modules
$esbuildPath = Join-Path -Path (Resolve-Path .).Path -ChildPath "frontend\node_modules\@esbuild\win32-x64\esbuild.exe"
if (Test-Path $esbuildPath) {
  try {
    Write-Host "Removing $esbuildPath"
    Remove-Item -Path $esbuildPath -Force -ErrorAction Stop
  } catch {
  Write-Warning "Could not remove ${esbuildPath}: $_"
  }
} else {
  Write-Host "esbuild binary not present at $esbuildPath"
}

# Run install (skip lifecycle scripts to avoid husky errors in detached environments)
Write-Host "Running npm ci --ignore-scripts in frontend..."
Push-Location frontend
try {
  npm ci --ignore-scripts
} catch {
  Write-Warning "npm ci failed: $_"
  Pop-Location
  exit 1
}

Write-Host "Running frontend tests (vitest)..."
try {
  npx vitest --run --reporter verbose
} catch {
  Write-Warning "vitest run failed: $_"
  Pop-Location
  exit 1
}

Pop-Location
Write-Host "Done. If tests passed, you can run 'npm ci' without --ignore-scripts later to run prepare." 
