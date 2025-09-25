from fastapi import APIRouter

router = APIRouter()


@router.get("")
async def list_trades():
    # Minimal response for tests
    return []
