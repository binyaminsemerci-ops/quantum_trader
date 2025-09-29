from fastapi import APIRouter
from config.config import load_config, masked_config_summary
from backend.utils.metrics import update_health_metric
from backend.routes.settings import SETTINGS

router = APIRouter()


@router.get("/health")
def health_check():
    try:
        cfg = load_config()
        summary = masked_config_summary(cfg)
        capabilities = {
            "exchanges": {
                "binance": bool(cfg.binance_api_key and cfg.binance_api_secret),
                "coinbase": bool(cfg.coinbase_api_key and cfg.coinbase_api_secret),
                "kucoin": bool(cfg.kucoin_api_key and cfg.kucoin_api_secret),
            },
            "live_market_data": bool(SETTINGS.get('ENABLE_LIVE_MARKET_DATA', getattr(cfg, 'enable_live_market_data', False))),
        }
        update_health_metric(True)
        return {"status": "ok", "secrets": summary, "capabilities": capabilities}
    except Exception:
        update_health_metric(False)
        raise
