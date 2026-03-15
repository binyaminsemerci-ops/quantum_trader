"""Journal Router — Trade journal entries CRUD"""
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from db.connection import get_db
from db.models import JournalEntry
from schemas import JournalEntryCreate, JournalEntryResponse
from auth.auth_router import verify_token, TokenData

router = APIRouter(prefix="/journal", tags=["Trade Journal"])


@router.get("/entries", response_model=list[JournalEntryResponse])
def list_entries(
    symbol: Optional[str] = Query(None),
    strategy: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """List journal entries with optional filters"""
    q = db.query(JournalEntry).order_by(JournalEntry.created_at.desc())
    if symbol:
        q = q.filter(JournalEntry.trade_symbol == symbol.upper())
    if strategy:
        q = q.filter(JournalEntry.strategy_tag == strategy)
    return q.offset(offset).limit(limit).all()


@router.get("/entries/{entry_id}", response_model=JournalEntryResponse)
def get_entry(entry_id: int, db: Session = Depends(get_db)):
    """Get a single journal entry"""
    entry = db.query(JournalEntry).filter(JournalEntry.id == entry_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Journal entry not found")
    return entry


@router.post("/entries", response_model=JournalEntryResponse)
def create_entry(
    payload: JournalEntryCreate,
    db: Session = Depends(get_db),
    token: TokenData = Depends(verify_token),
):
    """Create a new journal entry (requires auth)"""
    entry = JournalEntry(
        trade_symbol=payload.trade_symbol.upper(),
        trade_side=payload.trade_side,
        entry_price=payload.entry_price,
        exit_price=payload.exit_price,
        pnl=payload.pnl,
        strategy_tag=payload.strategy_tag,
        notes=payload.notes,
        rating=payload.rating,
        mistakes=payload.mistakes,
        lessons=payload.lessons,
        created_by=token.username,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


@router.put("/entries/{entry_id}", response_model=JournalEntryResponse)
def update_entry(
    entry_id: int,
    payload: JournalEntryCreate,
    db: Session = Depends(get_db),
    token: TokenData = Depends(verify_token),
):
    """Update a journal entry (requires auth)"""
    entry = db.query(JournalEntry).filter(JournalEntry.id == entry_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Journal entry not found")
    for field, value in payload.model_dump(exclude_unset=True).items():
        if field == "trade_symbol" and value:
            value = value.upper()
        setattr(entry, field, value)
    db.commit()
    db.refresh(entry)
    return entry


@router.delete("/entries/{entry_id}")
def delete_entry(
    entry_id: int,
    db: Session = Depends(get_db),
    token: TokenData = Depends(verify_token),
):
    """Delete a journal entry (admin only)"""
    if token.role != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")
    entry = db.query(JournalEntry).filter(JournalEntry.id == entry_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Journal entry not found")
    db.delete(entry)
    db.commit()
    return {"status": "deleted", "id": entry_id}


@router.get("/stats")
def journal_stats(db: Session = Depends(get_db)):
    """Aggregate journal statistics"""
    from sqlalchemy import func
    total = db.query(func.count(JournalEntry.id)).scalar() or 0
    avg_rating = db.query(func.avg(JournalEntry.rating)).filter(
        JournalEntry.rating.isnot(None)
    ).scalar()
    avg_pnl = db.query(func.avg(JournalEntry.pnl)).filter(
        JournalEntry.pnl.isnot(None)
    ).scalar()
    top_strategies = (
        db.query(JournalEntry.strategy_tag, func.count(JournalEntry.id))
        .filter(JournalEntry.strategy_tag.isnot(None))
        .group_by(JournalEntry.strategy_tag)
        .order_by(func.count(JournalEntry.id).desc())
        .limit(5)
        .all()
    )
    return {
        "total_entries": total,
        "avg_rating": round(avg_rating, 2) if avg_rating else None,
        "avg_pnl": round(avg_pnl, 4) if avg_pnl else None,
        "top_strategies": [{"tag": t, "count": c} for t, c in top_strategies],
    }
