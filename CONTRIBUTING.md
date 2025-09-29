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

- Backend tests: `python -m pytest -q`.
- Frontend tests (if touched): `npm run test` / `npm run lint`.
- Linters: `python -m ruff check backend` and `python -m mypy backend`.
- Run `pre-commit run --all-files`.
- No secrets checked into repo (see `.env.example`).
- If runtime deps change, update `backend/requirements.txt` / `backend/requirements-dev.txt` and mention it in the PR.

Review

- At least one reviewer should approve major changes. For security-sensitive
  changes (secrets, model loading, order execution), request a second reviewer.

Security policy

- Never commit API keys or credentials. Use environment variables (see `.env.example`).

Thanks — contributions are appreciated!

## Local setup

### Backend
1. Create a virtual environment (`python -m venv .venv`).
2. Activate it and install runtime deps: `pip install -r backend/requirements.txt`.
3. Install dev tooling when you need it: `pip install -r backend/requirements-dev.txt`.
4. Run the API with `uvicorn backend.main:app --reload --port 8000`.

### Frontend
1. `cd frontend`
2. `npm install`
3. `npm run dev` (Vite dev server on <http://localhost:5173>).

### Tooling
- Install pre-commit (`pip install pre-commit`) and run `pre-commit install` once.
- Before pushing, run `pre-commit run --all-files` to avoid CI lint failures.
- Stress harness tests: `python scripts/stress/harness.py --count 1` and
  `pytest scripts/stress/tests`.
