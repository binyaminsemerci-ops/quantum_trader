from fastapi import APIRouter
from backend.utils.trade_logger import get_trades

router = APIRouter()

@router.get("/")
def list_trades():
    return get_trades()
