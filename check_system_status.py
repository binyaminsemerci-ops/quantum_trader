#!/usr/bin/env python3
"""
Oppdatert systemstatus etter Node.js-bekreftelse
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path


def check_system_status():
    """Sjekk gjeldende systemstatus."""

    print("🔍 Sjekker komplett systemstatus...")

    status = {
        "timestamp": datetime.now().isoformat(),
        "components": {},
        "services": {},
        "frontend": {},
        "backend": {},
    }

    # Sjekk komponenter
    print("📁 Sjekker systemkomponenter...")
    status["components"] = {
        "backend_dir": Path("backend").exists(),
        "frontend_dir": Path("frontend").exists(),
        "database": Path("backend/quantum_trader.db").exists(),
        "package_json": Path("frontend/package.json").exists(),
        "vite_config": Path("frontend/vite.config.tsx").exists(),
        "backend_main": Path("backend/simple_main.py").exists(),
    }

    # Sjekk Node.js og npm
    print("🛠️ Sjekker Node.js og npm...")
    try:
        node_result = subprocess.run(
            ["node", "--version"], capture_output=True, text=True, timeout=5
        )
        npm_result = subprocess.run(
            ["npm", "--version"], capture_output=True, text=True, timeout=5
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
    print("📦 Sjekker frontend-avhengigheter...")
    try:
        if Path("frontend/node_modules").exists():
            npm_ls_result = subprocess.run(
                ["npm", "ls", "--depth=0"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd="frontend",
            )
            status["frontend"]["dependencies_installed"] = npm_ls_result.returncode == 0
            status["frontend"]["node_modules_exists"] = True
        else:
            status["frontend"]["dependencies_installed"] = False
            status["frontend"]["node_modules_exists"] = False
    except Exception as e:
        status["frontend"]["dependencies_error"] = str(e)

    # Test frontend build
    print("🏗️ Tester frontend build...")
    try:
        build_result = subprocess.run(
            ["npm", "run", "build"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd="frontend",
        )
        status["frontend"]["can_build"] = build_result.returncode == 0
        if build_result.returncode != 0:
            status["frontend"]["build_error"] = build_result.stderr[:500]
    except Exception as e:
        status["frontend"]["can_build"] = False
        status["frontend"]["build_error"] = str(e)

    # Sjekk tjenester (uten å starte dem)
    print("🌐 Sjekker tjenestestatus...")
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


def print_status_report(status):
    """Print formatert statusrapport."""

    print("\n" + "=" * 60)
    print("📊 QUANTUM TRADER - SYSTEMSTATUS")
    print("=" * 60)

    # Komponenter
    components = status["components"]
    print("\n🏗️ SYSTEMKOMPONENTER:")
    print(f"   Backend mappe: {'✅' if components['backend_dir'] else '❌'}")
    print(f"   Frontend mappe: {'✅' if components['frontend_dir'] else '❌'}")
    print(f"   Database: {'✅' if components['database'] else '❌'}")
    print(f"   Package.json: {'✅' if components['package_json'] else '❌'}")
    print(f"   Vite config: {'✅' if components['vite_config'] else '❌'}")
    print(f"   Backend main: {'✅' if components['backend_main'] else '❌'}")

    # Frontend-miljø
    frontend = status["frontend"]
    print("\n🛠️ FRONTEND-MILJØ:")
    if frontend.get("node_available"):
        print(f"   ✅ Node.js: {frontend.get('node_version', 'ukjent versjon')}")
    else:
        print("   ❌ Node.js: Ikke tilgjengelig")

    if frontend.get("npm_available"):
        print(f"   ✅ npm: {frontend.get('npm_version', 'ukjent versjon')}")
    else:
        print("   ❌ npm: Ikke tilgjengelig")

    print(
        f"   Dependencies: {'✅ Installert' if frontend.get('dependencies_installed') else '❌ Mangler'}"
    )
    print(f"   Kan bygge: {'✅ Ja' if frontend.get('can_build') else '❌ Nei'}")

    # Tjenester
    services = status["services"]
    print("\n🌐 TJENESTER:")
    print(
        f"   Backend (port 8000): {'✅ Kjører' if services['backend_running'] else '❌ Ikke kjører'}"
    )
    print(
        f"   Frontend (port 5173): {'✅ Kjører' if services['frontend_running'] else '❌ Ikke kjører'}"
    )

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

    services_running = len([v for v in services.values() if v])

    print("\n📈 SAMLET VURDERING:")
    print(
        f"   Komponenter: {component_score:.1%} ({total_components}/{total_possible})"
    )
    print(f"   Frontend-miljø: {frontend_score:.1%}")
    print(f"   Tjenester kjører: {services_running}/2")

    overall_score = (component_score + frontend_score) / 2

    if overall_score >= 0.8:
        print(f"   🟢 Status: UTMERKET ({overall_score:.1%})")
        print("   💡 System er klar for utvikling og testing!")
    elif overall_score >= 0.6:
        print(f"   🟡 Status: BRA ({overall_score:.1%})")
        print("   💡 System er i hovedsak funksjonelt")
    else:
        print(f"   🔴 Status: TRENGER ARBEID ({overall_score:.1%})")
        print("   💡 Flere komponenter må fikses")

    print("\n" + "=" * 60)


def main():
    """Hovedfunksjon."""

    # Sjekk status
    status = check_system_status()

    # Print rapport
    print_status_report(status)

    # Lagre til fil
    with open("current_system_status.json", "w") as f:
        json.dump(status, f, indent=2, default=str)

    print("📄 Detaljert status lagret til: current_system_status.json")

    return 0


if __name__ == "__main__":
    exit(main())
