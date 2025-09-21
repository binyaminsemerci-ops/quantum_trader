from fastapi import APIRouter
from typing import Any

router = APIRouter()

SETTINGS: dict[str, Any] = {}


@router.get("")
async def get_settings():
    return SETTINGS


@router.post("")
async def post_settings(payload: dict):
    SETTINGS.update(payload)
    return {"status": "ok", "settings": SETTINGS}
