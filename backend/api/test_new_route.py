"""Test route to verify new routes can be registered."""
from fastapi import APIRouter

router = APIRouter(prefix="/api/test123", tags=["test123"])

@router.get("/hello")
async def test_hello():
    return {"message": "Hello from test123!", "status": "success"}
