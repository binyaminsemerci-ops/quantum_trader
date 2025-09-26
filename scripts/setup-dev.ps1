param(
  [string]$VenvDir = '.venv'
)

Write-Host "Creating virtualenv in $VenvDir (if missing)"
if (-not (Test-Path $VenvDir)) {
  python -m venv $VenvDir
}

Write-Host "Activating virtualenv and installing dev requirements"
$activate = Join-Path $VenvDir 'Scripts\Activate.ps1'
if (Test-Path $activate) {
  & $activate
}

$req = 'backend/requirements-dev.txt'
if (Test-Path $req) {
  python -m pip install --upgrade pip
  pip install -r $req
} else {
  Write-Host "No $req found; skipping dev dependency install"
}

Write-Host "Configuring git hooks path to .githooks"
git config core.hooksPath .githooks

Write-Host "Dev setup complete. Activate the venv with: .\\.venv\\Scripts\\Activate.ps1"
