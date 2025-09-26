from fastapi import APIRouter
from config.config import load_config, masked_config_summary

router = APIRouter()


@router.get("/health")
def health_check():
    cfg = load_config()
    summary = masked_config_summary(cfg)
    # capability summary - which adapters are available (based on presence of keys)
    capabilities = {
        "exchanges": {
            "binance": bool(cfg.binance_api_key and cfg.binance_api_secret),
            "coinbase": bool(cfg.coinbase_api_key and cfg.coinbase_api_secret),
            "kucoin": bool(cfg.kucoin_api_key and cfg.kucoin_api_secret),
        }
    }
    return {"status": "ok", "secrets": summary, "capabilities": capabilities}
