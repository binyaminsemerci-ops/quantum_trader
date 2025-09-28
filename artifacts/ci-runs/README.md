This folder stores downloaded CI run artifacts gathered by `scripts/ci-watch.ps1`.

Usage

- Run the watcher once (requires GitHub CLI authenticated):

```powershell
pwsh .\scripts\ci-watch.ps1 -Once
```

- Continuous polling (every 60s):

```powershell
pwsh .\scripts\ci-watch.ps1
```

Artifacts are saved under `artifacts/ci-runs/<run-id>/` with a `metadata.json` file.
