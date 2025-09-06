from fastapi import APIRouter

router = APIRouter()

@router.get("")
async def get_trades():
    return [{"id": 1, "symbol": "BTCUSDT", "side": "BUY"}]
