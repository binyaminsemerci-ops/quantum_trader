from fastapi import APIRouter

router = APIRouter()


@router.get("")
async def stats():
    return {"total_trades": 0}
