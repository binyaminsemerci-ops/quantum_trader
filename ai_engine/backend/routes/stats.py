from fastapi import APIRouter
from backend.utils.trade_logger import get_balance_and_pnl

router = APIRouter()


@router.get("/")
def get_stats():
    return get_balance_and_pnl()
