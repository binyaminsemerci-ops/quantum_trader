#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
AUTO_YES=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift;;
    --yes|-y) AUTO_YES=1; shift;;
    *) echo "Unknown arg: $1"; exit 2;;
  esac
done

if [ ! -f backend/dev_in_runtime.txt ]; then
  echo "No backend/dev_in_runtime.txt found. Run the check first: python backend/scripts/check_dev_deps_in_runtime.py"
  exit 0
fi

pkgs=$(cat backend/dev_in_runtime.txt | tr ',' ' ')
echo "The following dev-only packages are detected in your runtime environment: $pkgs"

if [ "$DRY_RUN" -eq 1 ]; then
  echo "Dry-run: would uninstall: $pkgs"
  exit 0
fi

if [ "$AUTO_YES" -eq 1 ]; then
  echo "Auto-confirm enabled; uninstalling: $pkgs"
  python -m pip uninstall -y $pkgs
else
  read -p "Do you want to uninstall these packages now? [y/N] " yn
  case "$yn" in
    [Yy]* ) python -m pip uninstall -y $pkgs;;
    * ) echo "Aborted. No changes made."; exit 0;;
  esac
fi

echo "Uninstalled: $pkgs"
exit 0
