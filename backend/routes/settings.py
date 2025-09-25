from fastapi import APIRouter

router = APIRouter()


@router.get("")
async def get_settings():
    return {"api_key": "dummy"}


@router.post("")
async def save_settings(payload: dict):
    return {"status": "ok"}
