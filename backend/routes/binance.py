from fastapi import APIRouter
from typing import Optional
from config.config import load_config
from backend.utils.exchanges import get_exchange_client

router = APIRouter()


@router.get("/server-time")
async def server_time():
    return {"serverTime": 1234567890}


def _pick_credentials_for_exchange(cfg, exchange: Optional[str]):
    exch = (exchange or '').lower()
    if exch == 'coinbase':
        return cfg.coinbase_api_key, cfg.coinbase_api_secret
    if exch == 'kucoin':
        return cfg.kucoin_api_key, cfg.kucoin_api_secret
    # default to binance
    return cfg.binance_api_key, cfg.binance_api_secret


@router.get("/spot-balance")
async def spot_balance(exchange: Optional[str] = None):
    cfg = load_config()
    api_key, api_secret = _pick_credentials_for_exchange(cfg, exchange)
    client = get_exchange_client(name=exchange, api_key=api_key, api_secret=api_secret)
    return client.spot_balance()


@router.get("/futures-balance")
async def futures_balance(exchange: Optional[str] = None):
    cfg = load_config()
    api_key, api_secret = _pick_credentials_for_exchange(cfg, exchange)
    client = get_exchange_client(name=exchange, api_key=api_key, api_secret=api_secret)
    return client.futures_balance()
