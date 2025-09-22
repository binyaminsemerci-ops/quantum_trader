#!/usr/bin/env bash
set -euo pipefail

if [ ! -f backend/dev_in_runtime.txt ]; then
  echo "No backend/dev_in_runtime.txt found. Run the check first: python backend/scripts/check_dev_deps_in_runtime.py"
  exit 0
fi

pkgs=$(cat backend/dev_in_runtime.txt | tr ',' ' ')
echo "The following dev-only packages are detected in your runtime environment: $pkgs"
read -p "Do you want to uninstall these packages now? [y/N] " yn
case "$yn" in
  [Yy]* ) python -m pip uninstall -y $pkgs;;
  * ) echo "Aborted. No changes made."; exit 0;;
esac

echo "Uninstalled: $pkgs"
exit 0
