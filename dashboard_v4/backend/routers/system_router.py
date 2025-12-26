"""System Health Monitoring and Auto-Healing Endpoints"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from auth.auth_router import verify_token, TokenData
from db.connection import get_db
from db.models import SystemEvent
from datetime import datetime, timedelta
import psutil
import time
import subprocess

router = APIRouter(prefix="/system", tags=["System Health & Monitoring"])


def log_event(db: Session, event: str, cpu: float, ram: float, disk: float = None, 
              details: str = None, severity: str = "info"):
    """Log system event to database"""
    try:
        log_entry = SystemEvent(
            event=event,
            cpu=cpu,
            ram=ram,
            disk=disk,
            details=details,
            severity=severity,
            timestamp=datetime.utcnow()
        )
        db.add(log_entry)
        db.commit()
        db.refresh(log_entry)
        return log_entry
    except Exception as e:
        print(f"⚠️ Failed to log system event: {e}")
        db.rollback()
        return None


def get_system_metrics():
    """Get current system hardware metrics"""
    try:
        cpu = psutil.cpu_percent(interval=1)
        ram = psutil.virtual_memory().percent
        disk = psutil.disk_usage('/').percent
        uptime = time.time() - psutil.boot_time()
        
        return {
            "cpu": round(cpu, 1),
            "ram": round(ram, 1),
            "disk": round(disk, 1),
            "uptime_sec": int(uptime),
            "uptime_hours": round(uptime / 3600, 1)
        }
    except Exception as e:
        print(f"⚠️ Failed to get system metrics: {e}")
        return {"cpu": 0.0, "ram": 0.0, "disk": 0.0, "uptime_sec": 0, "uptime_hours": 0.0, "error": str(e)}


def get_container_status():
    """Get Docker container status"""
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}:{{.Status}}"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            containers = {}
            for line in result.stdout.strip().split('\n'):
                if ':' in line:
                    name, status_str = line.split(':', 1)
                    containers[name] = status_str.strip()
            return containers
        return {}
    except Exception as e:
        print(f"⚠️ Failed to get container status: {e}")
        return {}


@router.get("/health")
def system_health(db: Session = Depends(get_db)):
    """Get current system health metrics - Public endpoint"""
    metrics = get_system_metrics()
    containers = get_container_status()
    
    cpu = metrics.get("cpu", 0)
    ram = metrics.get("ram", 0)
    
    if cpu > 90 or ram > 90:
        health_status = "CRITICAL"
        severity = "critical"
    elif cpu > 75 or ram > 75:
        health_status = "STRESSED"
        severity = "warning"
    else:
        health_status = "HEALTHY"
        severity = "info"
    
    log_event(db, "health_check", cpu, ram, metrics.get("disk", 0), f"Status: {health_status}", severity)
    
    return {
        "status": health_status,
        "metrics": metrics,
        "containers": containers,
        "container_count": len(containers),
        "timestamp": datetime.utcnow().isoformat()
    }


@router.post("/self_heal")
def self_heal(token_data: TokenData = Depends(verify_token), db: Session = Depends(get_db)):
    """Trigger system self-healing - Admin only"""
    if token_data.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin privileges required")
    
    metrics = get_system_metrics()
    cpu = metrics.get("cpu", 0)
    ram = metrics.get("ram", 0)
    
    if cpu < 85 and ram < 85:
        log_event(db, "self_heal_check", cpu, ram, details="No healing required", severity="info")
        return {"status": "no_action_needed", "message": "System healthy", "metrics": metrics, "user": token_data.username}
    
    healing_actions = []
    try:
        container_name = "quantum_dashboard_backend"
        result = subprocess.run(["docker", "restart", container_name], capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            healing_actions.append(f"Restarted {container_name}")
        else:
            healing_actions.append(f"Failed: {result.stderr}")
    except Exception as e:
        healing_actions.append(f"Error: {str(e)}")
    
    log_event(db, "self_heal_executed", cpu, ram, details=f"Actions: {', '.join(healing_actions)}", severity="warning")
    
    return {"status": "healing_completed", "actions": healing_actions, "metrics": metrics, "user": token_data.username}


@router.post("/restart_container")
def restart_container(container: str, token_data: TokenData = Depends(verify_token), db: Session = Depends(get_db)):
    """Restart specific container - Admin only"""
    if token_data.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin privileges required")
    
    allowed = ["quantum_dashboard_backend", "quantum_dashboard_frontend", "quantum_postgres", "quantum_redis"]
    if container not in allowed:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Must be one of: {', '.join(allowed)}")
    
    metrics = get_system_metrics()
    try:
        result = subprocess.run(["docker", "restart", container], capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            log_event(db, "container_restart", metrics.get("cpu", 0), metrics.get("ram", 0), 
                     details=f"{container} restarted by {token_data.username}", severity="warning")
            return {"status": "success", "message": f"{container} restarted", "user": token_data.username}
        raise HTTPException(status_code=500, detail=f"Failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/events")
def get_system_events(limit: int = 50, severity: str = None, token_data: TokenData = Depends(verify_token), db: Session = Depends(get_db)):
    """Get system events log - All authenticated users"""
    try:
        query = db.query(SystemEvent)
        if severity:
            query = query.filter(SystemEvent.severity == severity)
        events = query.order_by(SystemEvent.timestamp.desc()).limit(limit).all()
        return {
            "total": len(events),
            "events": [{
                "id": e.id, "event": e.event, "cpu": e.cpu, "ram": e.ram, "disk": e.disk,
                "details": e.details, "severity": e.severity,
                "timestamp": e.timestamp.isoformat() if e.timestamp else None
            } for e in events]
        }
    except Exception:
        return {"error": "Database not connected", "events": []}


@router.get("/metrics/history")
def metrics_history(hours: int = 24, token_data: TokenData = Depends(verify_token), db: Session = Depends(get_db)):
    """Get historical metrics time-series"""
    try:
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        events = db.query(SystemEvent).filter(
            SystemEvent.timestamp >= cutoff, SystemEvent.event == "health_check"
        ).order_by(SystemEvent.timestamp).all()
        return {
            "period_hours": hours,
            "data_points": len(events),
            "metrics": [{"timestamp": e.timestamp.isoformat(), "cpu": e.cpu, "ram": e.ram, "disk": e.disk} for e in events]
        }
    except Exception:
        return {"error": "Failed to retrieve history", "metrics": []}
