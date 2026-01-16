#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/deploy-vps.sh [--build]
#
# Pre-req on VPS (Ubuntu):
#   systemd services already configured
#   Services running as quantum-*.service units
#
# Prepare env:
#   cp backend/.env.live.example backend/.env.live
#   # edit backend/.env.live and fill secrets

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. && pwd)
cd "$ROOT_DIR"

BUILD_FLAG=""
if [[ "${1:-}" == "--build" ]]; then
  BUILD_FLAG="--build"
fi

echo "[deploy] Restarting systemd services"
sudo systemctl restart 'quantum-*.service'

echo "[deploy] Services status:" 
systemctl list-units 'quantum-*.service' --no-pager --no-legend | head -10

echo "[deploy] Health check (frontend):"
set +e
curl -fsS http://localhost/ || true
set -e

echo "[deploy] Done. Ensure DNS points to this VPS and open ports 80/443 as needed."
