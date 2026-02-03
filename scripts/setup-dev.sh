#!/usr/bin/env bash
set -euo pipefail

# Minimal dev setup script (POSIX):
# - create a virtualenv in .venv
# - install dev requirements
# - configure local git hooks to use .githooks

VE_DIR=.venv
REQ_DEV=backend/requirements-dev.txt

echo "Creating virtualenv in ${VE_DIR} (if missing)"
if [ ! -d "${VE_DIR}" ]; then
  python -m venv "${VE_DIR}"
fi

echo "Activating virtualenv and installing dev requirements"
# shellcheck disable=SC1091
source "${VE_DIR}/bin/activate"
if [ -f "$REQ_DEV" ]; then
  python -m pip install --upgrade pip
  pip install -r "$REQ_DEV"
else
  echo "No $REQ_DEV found; skipping dev dependency install"
fi

echo "Configuring git hooks path to .githooks"
git config core.hooksPath .githooks || true

echo "Dev setup complete. Activate the venv with: source ${VE_DIR}/bin/activate"
