# backend — developer quickstart

This file documents the minimal steps for setting up the Python backend for
local development and testing.

Install runtime requirements (for running the app):

```pwsh
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
```

Install developer/test requirements (for running tests, linters, and local
tools):

```pwsh
pip install -r backend/requirements-dev.txt
```

## Why dev-only for some packages
We intentionally keep some packages (for example `SQLAlchemy-Utils`) in
`backend/requirements-dev.txt` rather than runtime requirements. This avoids
installing developer-only tooling in CI/runtime, reduces the attack surface,
and ensures security advisories are tracked and addressed explicitly.

If you need the dev tools locally, run the command above. CI intentionally
installs only specific test/lint/security tools so runtime environments stay
minimal.

## Check for accidental dev-only installs
A small script is provided to help detect if any dev-only packages are
present in your runtime environment (useful for pre-commit checks or local
validation):

```pwsh
python backend/scripts/check_dev_deps_in_runtime.py
```

If it prints a list of packages, you may have installed dev requirements into
your runtime environment. CI runs this script and emits a non-blocking warning
if any dev-only packages are detected.

## Enable local git pre-commit hook (optional)
To enable the included local git hook that prevents commits when dev-only
packages are present in your runtime environment:

```pwsh
# From repo root (one-time):
git config core.hooksPath .githooks
```

After that, the `.githooks/pre-commit` script will run on each commit and abort
the commit if dev-only packages are detected.

## Makefile target
You can also run the check locally via the Makefile target from the repo root:

```pwsh
make -C backend check-dev-deps
```

## Windows / PowerShell notes
Windows developers can use PowerShell to set up and run the same tools:

```powershell
# Create and activate the venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dev deps
pip install -r backend/requirements-dev.txt

# Run the check
python backend/scripts/check_dev_deps_in_runtime.py
```

## Repair helper
If the check finds dev-only packages installed at runtime, you can run the
repair helper to uninstall them (it will prompt for confirmation):

POSIX:
```bash
./scripts/repair-dev-deps.sh
```

PowerShell:
```powershell
.\scripts\repair-dev-deps.ps1
```

Both scripts support a dry-run mode to preview what would be uninstalled:

POSIX:
```bash
./scripts/repair-dev-deps.sh --dry-run
```

PowerShell:
```powershell
.\scripts\repair-dev-deps.ps1 -DryRun
```

## Using an isolated linters virtualenv locally

The CI uses an isolated `.venv_linters` virtualenv to install linters and
security scanners so their transitive dependencies do not appear in the
application runtime environment (and therefore do not trigger the dev-deps
enforcement check).

If you'd like to replicate CI locally, create the linters venv with:

```powershell
python -m venv .venv_linters --system-site-packages
.\.venv_linters\Scripts\Activate.ps1
.venv_linters\Scripts\pip install --upgrade pip
.venv_linters\Scripts\pip install ruff mypy black bandit safety
```


Notes:

- `--system-site-packages` lets tools in the linters venv import your
  runtime packages without reinstalling them, which prevents false
  "import not found" errors in mypy while keeping the linters' own deps
  isolated from your runtime Python.

- Do not install test-only packages (pytest, pytest-asyncio, etc.) into the
  runtime interpreter used by the app; CI installs those into the runner
  Python only after the enforcement check.

## Environment variables

For local development copy `backend/.env.example` to `backend/.env` and set
your local values (do not commit `backend/.env`). The repo includes a
`backend/.env.example` file with common values to get started.

Note: the example now defaults to a local SQLite database so a username
or external DB credentials are not required for standalone local runs or
for running the test-suite. If you plan to run the application against an
external Postgres instance, update `DB_USER`, `DB_PASS` and `DATABASE_URL`
in your local `backend/.env` accordingly.


- `--system-site-packages` lets tools in the linters venv import your
  runtime packages without reinstalling them, which prevents false
  "import not found" errors in mypy while keeping the linters' own deps
  isolated from your runtime Python.

- Do not install test-only packages (pytest, pytest-asyncio, etc.) into the
  runtime interpreter used by the app; CI installs those into the runner
  Python only after the enforcement check.
