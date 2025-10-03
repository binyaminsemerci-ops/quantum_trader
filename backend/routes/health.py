from fastapi import APIRouter

from backend.routes.settings import SETTINGS
from backend.utils.metrics import update_health_metric
# Defensive import for config with fallback shim
try:  # pragma: no cover
    from config.config import load_config, masked_config_summary  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    def load_config():  # type: ignore
        class _Cfg:
            binance_api_key = None
            binance_api_secret = None
            coinbase_api_key = None
            coinbase_api_secret = None
            kucoin_api_key = None
            kucoin_api_secret = None
            enable_live_market_data = False

        return _Cfg()

    def masked_config_summary(_cfg):  # type: ignore
        return {
            "has_binance_keys": False,
            "has_coinbase_keys": False,
        }

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
            "live_market_data": bool(
                SETTINGS.get(
                    "ENABLE_LIVE_MARKET_DATA",
                    getattr(cfg, "enable_live_market_data", False),
                ),
            ),
        }
        update_health_metric(True)
        return {"status": "ok", "secrets": summary, "capabilities": capabilities}
    except Exception:
        update_health_metric(False)
        raise
