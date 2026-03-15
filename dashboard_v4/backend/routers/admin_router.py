"""Admin Router — Consolidated admin dashboard endpoints"""
from fastapi import APIRouter, Depends, HTTPException
from auth.auth_router import verify_token, TokenData, USERS
import subprocess
import psutil
import os
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["Admin Dashboard"])


def _require_admin(token: TokenData = Depends(verify_token)) -> TokenData:
    if token.role != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")
    return token


@router.get("/users")
def list_users(token: TokenData = Depends(_require_admin)):
    """List all users (no passwords)"""
    return [
        {"username": u, "role": d["role"]}
        for u, d in USERS.items()
    ]


@router.get("/services")
def list_services(token: TokenData = Depends(_require_admin)):
    """List all quantum-* systemd services and their statuses"""
    try:
        result = subprocess.run(
            ["systemctl", "list-units", "quantum-*.service", "--no-legend", "--all"],
            capture_output=True, text=True, timeout=10
        )
        services = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 4:
                services.append({
                    "name": parts[0],
                    "load": parts[1],
                    "active": parts[2],
                    "sub": parts[3],
                    "description": " ".join(parts[4:]) if len(parts) > 4 else "",
                })
        return {"services": services, "count": len(services)}
    except Exception as e:
        return {"services": [], "error": str(e)}


@router.post("/services/{service_name}/restart")
def restart_service(
    service_name: str,
    token: TokenData = Depends(_require_admin),
):
    """Restart a specific quantum service"""
    if not service_name.startswith("quantum-"):
        raise HTTPException(status_code=400, detail="Can only restart quantum-* services")
    try:
        result = subprocess.run(
            ["systemctl", "restart", service_name],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            return {"status": "failed", "error": result.stderr.strip()}
        return {"status": "restarted", "service": service_name}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@router.get("/services/{service_name}/logs")
def service_logs(
    service_name: str,
    lines: int = 50,
    token: TokenData = Depends(_require_admin),
):
    """Get recent journal logs for a service"""
    if not service_name.startswith("quantum-"):
        raise HTTPException(status_code=400, detail="Can only view quantum-* service logs")
    try:
        result = subprocess.run(
            ["journalctl", "-u", service_name, "-n", str(min(lines, 200)), "--no-pager"],
            capture_output=True, text=True, timeout=10
        )
        return {"service": service_name, "lines": result.stdout.strip().split("\n")}
    except Exception as e:
        return {"service": service_name, "error": str(e)}


@router.get("/system")
def system_overview(token: TokenData = Depends(_require_admin)):
    """Comprehensive system overview"""
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.5),
        "cpu_count": psutil.cpu_count(),
        "ram_total_gb": round(mem.total / (1024**3), 2),
        "ram_used_gb": round(mem.used / (1024**3), 2),
        "ram_percent": mem.percent,
        "disk_total_gb": round(disk.total / (1024**3), 2),
        "disk_used_gb": round(disk.used / (1024**3), 2),
        "disk_percent": round(disk.used / disk.total * 100, 1),
        "boot_time": psutil.boot_time(),
        "load_avg": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
    }
