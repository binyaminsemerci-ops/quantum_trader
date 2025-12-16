#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/deploy-vps.sh [--build]
#
# Pre-req on VPS (Ubuntu):
#   sudo apt-get update -y
#   sudo apt-get install -y docker.io docker-compose-plugin
#   sudo usermod -aG docker $USER && newgrp docker
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

echo "[deploy] Using live profile with VPS override"
docker compose --profile live -f docker-compose.yml -f docker-compose.vps.yml up -d $BUILD_FLAG

echo "[deploy] Services status:" 
docker compose ps

echo "[deploy] Health check (frontend):"
set +e
curl -fsS http://localhost/ || true
set -e

echo "[deploy] Done. Ensure DNS points to this VPS and open ports 80/443 as needed."
