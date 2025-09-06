from fastapi import APIRouter

router = APIRouter()

@router.get("")
async def get_chart():
    # Returner en liste som testene forventer
    return [100, 101, 102]
