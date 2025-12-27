<#
.\scripts\local_ci.ps1 - Runs local linters, type checks, tests and optionally builds/pushes the Docker image.

Usage examples:
  # Run linters, mypy and tests
  .\scripts\local_ci.ps1 -RunLint -RunTypeCheck -RunTests

  # Build Docker image only
  .\scripts\local_ci.ps1 -BuildDocker

  # Build and push (requires GHCR env vars set, see .env.example)
  .\scripts\local_ci.ps1 -BuildDocker -PushDocker

Notes:
- Assumes Python virtualenv is available in .venv (optional). If not, run from your preferred environment.
- Assumes docker CLI is installed for building/pushing images.
- To push to GHCR, set environment variables (or source a .env file):
   GHCR_NAMESPACE (org or username)
   GHCR_USERNAME (username for ghcr login; defaults to GHCR_NAMESPACE)
   GHCR_PAT (personal access token with write:packages/read:packages)

#>

param(
    [switch]$RunLint,
    [switch]$RunTypeCheck,
    [switch]$RunTests,
    [switch]$BuildDocker,
    [switch]$PushDocker
)

function Try-ActivateVenv {
    $repoRoot = Resolve-Path "$PSScriptRoot\.." | Select-Object -ExpandProperty Path
    $activate = Join-Path $repoRoot '.venv\Scripts\Activate.ps1'
    if (Test-Path $activate) {
        Write-Host "Activating virtualenv at $activate"
        . $activate
    } else {
        Write-Host "No .venv found at $activate - proceeding with current Python environment"
    }
}

Push-Location $PSScriptRoot
Try {
    Try-ActivateVenv

    if ($RunLint) {
        Write-Host "Running ruff..."
        python -m ruff check .
        Write-Host "Running black (check)..."
        python -m black --check .
    }

    if ($RunTypeCheck) {
        Write-Host "Running mypy..."
        python -m mypy .
    }

    if ($RunTests) {
        Write-Host "Running pytest..."
        python -m pytest -q
    }

    if ($BuildDocker) {
        $namespace = $env:GHCR_NAMESPACE
        if (-not $namespace) { $namespace = Read-Host 'Enter GHCR namespace (org or username)' }
        $tag = "local"
        $backendPath = Resolve-Path "$PSScriptRoot\..\backend"
        $dockerfile = Join-Path $backendPath 'Dockerfile'
        $image = "ghcr.io/$namespace/quantum_trader:$tag"
        Write-Host "Building Docker image: $image"
        docker build -t $image -f $dockerfile $backendPath

        if ($PushDocker) {
            $username = $env:GHCR_USERNAME
            if (-not $username) { $username = $namespace }
            $pat = $env:GHCR_PAT
            if (-not $pat) {
                throw "GHCR_PAT not set. Set environment variable GHCR_PAT or source a .env file with it."
            }
            Write-Host "Logging in to ghcr.io as $username"
            $pat | docker login ghcr.io -u $username --password-stdin
            Write-Host "Pushing image $image"
            docker push $image
        }
    }

    Write-Host "local_ci script completed"
} finally {
    Pop-Location
}
