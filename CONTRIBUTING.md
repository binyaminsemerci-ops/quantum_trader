<!-- CONTRIBUTING.md -->
# Contributing to Quantum Trader

Thanks for wanting to contribute! This short guide shows the workflow we
prefer and a few simple rules to keep the repository healthy.

Branching

- Work from `main`.

- Create feature branches named with the pattern `feat/<name>`, `fix/<name>` or `chore/<name>`.

Commits & PRs

- Keep commits focused and with meaningful messages.

- Open a pull request against `main` and include a short description of the
  change, testing steps, and any security considerations.

Pre-PR checklist

- All tests pass locally: `python -m pytest -q` (backend) and frontend tests if changed.

- Linters: `python -m ruff check backend` and `python -m mypy backend` for backend changes.

- No secrets checked into repo (see `.env.example`).

- If the change affects runtime deps, update `backend/requirements.txt` and note in the PR.

Review

- At least one reviewer should approve major changes. For security-sensitive
  changes (secrets, model loading, order execution), request a second reviewer.

Security policy

- Never commit API keys or credentials. Use environment variables (see `.env.example`).

Thanks — contributions are appreciated!
