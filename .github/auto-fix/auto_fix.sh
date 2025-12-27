#!/usr/bin/env bash
set -euo pipefail
echo "Running auto-fix script"

# Basic environment
python -m pip install --upgrade pip || true
if [ -f backend/requirements-dev.txt ]; then python -m pip install -r backend/requirements-dev.txt || true; fi
python -m pip install pre-commit ruff mypy || true

# Run pre-commit to auto-fix formatting etc.
pre-commit run --all-files || true

# Run ruff auto-fix
ruff check . --fix || true

# Run mypy on main packages
mypy -p backend -p ai_engine || true

# Run tests to ensure nothing broken
pytest -q || true

# If there are changes, commit and push them back to the branch that triggered the workflow
if [ -n "$(git status --porcelain)" ]; then
  git config user.name "github-actions[bot]"
  git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
  git add -A
  git commit -m "chore(auto-fix): apply automatic fixes (pre-commit/ruff/mypy)"
  # push back to current branch
  git push origin HEAD:$(git rev-parse --abbrev-ref HEAD)
else
  echo "No fixes to push"
fi

echo "Auto-fix script finished"
