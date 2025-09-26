from fastapi import APIRouter

router = APIRouter()


@router.get("/server-time")
async def server_time():
    return {"serverTime": 1234567890}


@router.get("/spot-balance")
async def spot_balance():
    return {"asset": "USDT", "free": 1000.0}


@router.get("/futures-balance")
async def futures_balance():
    return {"asset": "BTC", "balance": 0.5}
