Local CI helper
===============

There is a PowerShell helper script at `scripts/local_ci.ps1` to run linters, mypy, tests and to build/push the backend Docker image locally.

Usage (PowerShell):

```powershell
# Run linters, type checks and tests
.\scripts\local_ci.ps1 -RunLint -RunTypeCheck -RunTests

# Build Docker image only
.\scripts\local_ci.ps1 -BuildDocker

# Build and push image to GHCR (ensure GHCR_* env vars or .env are set)
.\scripts\local_ci.ps1 -BuildDocker -PushDocker
```

See `.env.example` for the GHCR variables to set when pushing.

