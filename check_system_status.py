#!/usr/bin/env python3
"""Oppdatert systemstatus etter Node.js-bekreftelse."""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def check_system_status():
    """Kontroller generelle systemstatus."""
    status = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "components": {},
        "services": {},
        "frontend": {},
        "backend": {},
    }

    # Sjekk komponenter
    status["components"] = {
        "backend_dir": Path("backend").exists(),
        "frontend_dir": Path("frontend").exists(),
        "database": Path("backend/data/trades.db").exists(),
        "package_json": Path("frontend/package.json").exists(),
        "vite_config": Path("frontend/vite.config.ts").exists(),
        "backend_main": Path("backend/main.py").exists(),
    }

    # Sjekk Node.js og npm
    try:
        node_result = subprocess.run(
            ["node", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        npm_result = subprocess.run(
            ["npm", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )

        status["frontend"]["node_version"] = (
            node_result.stdout.strip() if node_result.returncode == 0 else None
        )
        status["frontend"]["npm_version"] = (
            npm_result.stdout.strip() if npm_result.returncode == 0 else None
        )
        status["frontend"]["node_available"] = node_result.returncode == 0
        status["frontend"]["npm_available"] = npm_result.returncode == 0
    except Exception as e:
        status["frontend"]["error"] = str(e)

    # Sjekk frontend dependencies
    try:
        if Path("frontend/node_modules").exists():
            npm_ls_result = subprocess.run(
                ["npm", "ls", "--depth=0"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd="frontend",
                check=False,
            )
            status["frontend"]["dependencies_installed"] = npm_ls_result.returncode == 0
            status["frontend"]["node_modules_exists"] = True
        else:
            status["frontend"]["dependencies_installed"] = False
            status["frontend"]["node_modules_exists"] = False
    except Exception as e:
        status["frontend"]["dependencies_error"] = str(e)

    # Test frontend build
    try:
        build_result = subprocess.run(
            ["npm", "run", "build"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd="frontend",
            check=False,
        )
        status["frontend"]["can_build"] = build_result.returncode == 0
        if build_result.returncode != 0:
            status["frontend"]["build_error"] = build_result.stderr[:500]
    except Exception as e:
        status["frontend"]["can_build"] = False
        status["frontend"]["build_error"] = str(e)

    # Sjekk tjenester (uten å starte dem)
    try:
        import requests

        # Test om backend kjører
        backend_response = requests.get("http://localhost:8000/health", timeout=2)
        status["services"]["backend_running"] = backend_response.status_code == 200
    except Exception:
        status["services"]["backend_running"] = False

    try:
        # Test om frontend dev-server kjører
        frontend_response = requests.get("http://localhost:5173", timeout=2)
        status["services"]["frontend_running"] = frontend_response.status_code == 200
    except Exception:
        status["services"]["frontend_running"] = False

    return status


def print_status_report(status) -> None:
    """Print formatert statusrapport."""
    # Komponenter
    components = status["components"]

    # Frontend-miljø
    frontend = status["frontend"]
    if frontend.get("node_available"):
        pass
    else:
        pass

    if frontend.get("npm_available"):
        pass
    else:
        pass

    # Tjenester
    services = status["services"]

    # Samlet vurdering
    total_components = len([v for v in components.values() if v])
    total_possible = len(components)
    component_score = total_components / total_possible

    frontend_score = 0
    if frontend.get("node_available"):
        frontend_score += 0.3
    if frontend.get("npm_available"):
        frontend_score += 0.3
    if frontend.get("dependencies_installed"):
        frontend_score += 0.2
    if frontend.get("can_build"):
        frontend_score += 0.2

    len([v for v in services.values() if v])

    overall_score = (component_score + frontend_score) / 2

    if overall_score >= 0.8 or overall_score >= 0.6:
        pass
    else:
        pass


def main() -> int:
    """Hovedfunksjon."""
    # Sjekk status
    status = check_system_status()

    # Print rapport
    print_status_report(status)

    # Lagre til fil
    with open("current_system_status.json", "w") as f:
        json.dump(status, f, indent=2, default=str)

    return 0


if __name__ == "__main__":
    sys.exit(main())
