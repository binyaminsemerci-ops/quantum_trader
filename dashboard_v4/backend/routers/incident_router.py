"""Incident Router — Incident tracking CRUD"""
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from db.connection import get_db
from db.models import Incident
from schemas import IncidentCreate, IncidentUpdate, IncidentResponse
from auth.auth_router import verify_token, TokenData

router = APIRouter(prefix="/incidents", tags=["Incident Tracking"])


@router.get("/", response_model=list[IncidentResponse])
def list_incidents(
    status: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """List incidents with optional filters"""
    q = db.query(Incident).order_by(Incident.opened_at.desc())
    if status:
        q = q.filter(Incident.status == status)
    if severity:
        q = q.filter(Incident.severity == severity)
    if category:
        q = q.filter(Incident.category == category)
    return q.offset(offset).limit(limit).all()


@router.get("/stats")
def incident_stats(db: Session = Depends(get_db)):
    """Aggregate incident statistics"""
    from sqlalchemy import func
    total = db.query(func.count(Incident.id)).scalar() or 0
    open_count = db.query(func.count(Incident.id)).filter(
        Incident.status.in_(["open", "investigating"])
    ).scalar() or 0
    by_severity = dict(
        db.query(Incident.severity, func.count(Incident.id))
        .group_by(Incident.severity)
        .all()
    )
    by_category = dict(
        db.query(Incident.category, func.count(Incident.id))
        .filter(Incident.category.isnot(None))
        .group_by(Incident.category)
        .all()
    )
    return {
        "total": total,
        "open": open_count,
        "resolved": total - open_count,
        "by_severity": by_severity,
        "by_category": by_category,
    }


@router.get("/{incident_id}", response_model=IncidentResponse)
def get_incident(incident_id: int, db: Session = Depends(get_db)):
    """Get a single incident"""
    inc = db.query(Incident).filter(Incident.id == incident_id).first()
    if not inc:
        raise HTTPException(status_code=404, detail="Incident not found")
    return inc


@router.post("/", response_model=IncidentResponse)
def create_incident(
    payload: IncidentCreate,
    db: Session = Depends(get_db),
    token: TokenData = Depends(verify_token),
):
    """Create a new incident (requires auth)"""
    inc = Incident(
        title=payload.title,
        description=payload.description,
        severity=payload.severity,
        category=payload.category,
        affected_services=payload.affected_services,
        reported_by=token.username,
    )
    db.add(inc)
    db.commit()
    db.refresh(inc)
    return inc


@router.put("/{incident_id}", response_model=IncidentResponse)
def update_incident(
    incident_id: int,
    payload: IncidentUpdate,
    db: Session = Depends(get_db),
    token: TokenData = Depends(verify_token),
):
    """Update an incident (requires auth)"""
    inc = db.query(Incident).filter(Incident.id == incident_id).first()
    if not inc:
        raise HTTPException(status_code=404, detail="Incident not found")
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(inc, field, value)
    # Auto-set timestamps on status transitions
    if payload.status == "resolved" and inc.resolved_at is None:
        inc.resolved_at = datetime.utcnow()
    if payload.status == "closed" and inc.closed_at is None:
        inc.closed_at = datetime.utcnow()
    db.commit()
    db.refresh(inc)
    return inc


@router.delete("/{incident_id}")
def delete_incident(
    incident_id: int,
    db: Session = Depends(get_db),
    token: TokenData = Depends(verify_token),
):
    """Delete an incident (admin only)"""
    if token.role != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")
    inc = db.query(Incident).filter(Incident.id == incident_id).first()
    if not inc:
        raise HTTPException(status_code=404, detail="Incident not found")
    db.delete(inc)
    db.commit()
    return {"status": "deleted", "id": incident_id}
