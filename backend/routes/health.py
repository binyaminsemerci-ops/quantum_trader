from fastapi import APIRouter
from config.config import load_config, masked_config_summary
from backend.utils.metrics import update_health_metric

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
            }
        }
        update_health_metric(True)
        return {"status": "ok", "secrets": summary, "capabilities": capabilities}
    except Exception:
        update_health_metric(False)
        raise
