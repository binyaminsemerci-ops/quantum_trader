from fastapi import APIRouter

router = APIRouter()

SETTINGS = {}


@router.get("")
async def get_settings():
    return SETTINGS


@router.post("")
async def post_settings(payload: dict):
    SETTINGS.update(payload)
    return {"status": "ok", "settings": SETTINGS}
