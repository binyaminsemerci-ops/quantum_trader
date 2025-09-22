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

Why dev-only for some packages
------------------------------
We intentionally keep some packages (for example `SQLAlchemy-Utils`) in
`backend/requirements-dev.txt` rather than runtime requirements. This avoids
installing developer-only tooling in CI/runtime, reduces the attack surface,
and ensures security advisories are tracked and addressed explicitly.

If you need the dev tools locally, run the command above. CI intentionally
installs only specific test/lint/security tools so runtime environments stay
minimal.

Check for accidental dev-only installs
-------------------------------------
A small script is provided to help detect if any dev-only packages are
present in your runtime environment (useful for pre-commit checks or local
validation):

```pwsh
python backend/scripts/check_dev_deps_in_runtime.py
```

If it prints a list of packages, you may have installed dev requirements into
your runtime environment. CI runs this script and emits a non-blocking warning
if any dev-only packages are detected.

Enable local git pre-commit hook (optional)
-----------------------------------------
To enable the included local git hook that prevents commits when dev-only
packages are present in your runtime environment:

```pwsh
# From repo root (one-time):
git config core.hooksPath .githooks
```

After that, the `.githooks/pre-commit` script will run on each commit and abort
the commit if dev-only packages are detected.

Makefile target
----------------
You can also run the check locally via the Makefile target from the repo root:

```pwsh
make -C backend check-dev-deps
```

