#!/usr/bin/env bash
set -euo pipefail

echo "This script bootstraps the development environment." 

# Run POSIX setup
if [ -x "./scripts/setup-dev.sh" ]; then
  ./scripts/setup-dev.sh
else
  echo "Missing ./scripts/setup-dev.sh; aborting"
  exit 1
fi

read -p "Enable local git hooks (configure core.hooksPath to .githooks)? [y/N] " yn
case "$yn" in
  [Yy]* ) git config core.hooksPath .githooks; echo "Git hooks enabled.";;
  * ) echo "Skipping git hooks configuration.";;
esac

echo "Bootstrap complete. Activate the venv with: source .venv/bin/activate"
