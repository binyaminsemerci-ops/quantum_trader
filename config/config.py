
"""Runtime configuration using Pydantic settings.

The original project referenced a rich config layer; this updated module keeps
things lightweight but centralises access to environment variables. It provides
attribute-style access (``settings``) plus helpers used across the code base.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment and optional .env file."""

    default_exchange: str = Field(default="binance", alias="DEFAULT_EXCHANGE")
    default_quote: str = Field(default="USDT", alias="DEFAULT_QUOTE")
    futures_quote: str = Field(default="USDT", alias="FUTURES_QUOTE")

    default_symbols_raw: str | list[str] = Field(
        default="BTCUSDT,ETHUSDT,BNBUSDT,XRPUSDT,SOLUSDT,ADAUSDT,DOGEUSDT,TRXUSDT,DOTUSDT,LINKUSDT,AVAXUSDT,MATICUSDT,ATOMUSDT,LTCUSDT,UNIUSDT,ETCUSDT,XLMUSDT,NEARUSDT,AAVEUSDT,ALGOUSDT,APEUSDT,APTUSDT,ARBUSDT,AXSUSDT,FILUSDT,FLOWUSDT,FTMUSDT,GALAUSDT,HBARUSDT,ICPUSDT,IMXUSDT,INJUSDT,KAVAUSDT,KLAYUSDT,LDOUSDT,MKRUSDT,NEOUSDT,OPUSDT,QNTUSDT,RNDRUSDT,ROSEUSDT,RUNEUSDT,SANDUSDT,SNXUSDT,STXUSDT,SUIUSDT,THETAUSDT,TONUSDT,VETUSDT,XMRUSDT,XTZUSDT,ZILUSDT,ZRXUSDT,COMPUSDT,CRVUSDT,DYDXUSDT,ENSUSDT,GMTUSDT,GMXUSDT,HNTUSDT,IOSTUSDT,IOTAUSDT,KNCUSDT,KSMUSDT,LPTUSDT,LRCUSDT,MINAUSDT,NEXOUSDT,OCEANUSDT,OMGUSDT,ONEUSDT,RSRUSDT,SKLUSDT,SRMUSDT,STORJUSDT,SXPUSDT,VTHOUSDT,WAVESUSDT,XEMUSDT,ZECUSDT,BATUSDT,CHZUSDT,CELOUSDT,COTIUSDT,CTSIUSDT,DASHUSDT,ENJUSDT,FETUSDT,HOTUSDT,ICXUSDT,IOTXUSDT,JASMYUSDT,MANAUSDT,MTLUSDT,NKNUSDT,OGNUSDT,RLCUSDT,SCUSDT,STMXUSDT,XVGUSDT,ZENUSDT",
        alias="DEFAULT_SYMBOLS",
    )

    # Grouped symbol lists for convenience: main base coins, high-volume
    # Layer-1 coins and Layer-2 coins. These can be overridden via env vars
    # MAINBASE_SYMBOLS, LAYER1_SYMBOLS, LAYER2_SYMBOLS if needed.
    mainbase_symbols_raw: str | list[str] = Field(
        default="BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,ADAUSDT",
        alias="MAINBASE_SYMBOLS",
    )

    layer1_symbols_raw: str | list[str] = Field(
        default="ETHUSDT,BTCUSDT,SOLUSDT,AVAXUSDT,MATICUSDT,DOTUSDT,NEARUSDT,ADAUSDT",
        alias="LAYER1_SYMBOLS",
    )

    layer2_symbols_raw: str | list[str] = Field(
        default="OPUSDT,ARBUSDT,ARBUSDT,IMXUSDT,SKLUSDT,ENSUSDT,GMXUSDT",
        alias="LAYER2_SYMBOLS",
    )

    enable_live_market_data: bool = Field(default=False, alias="ENABLE_LIVE_MARKET_DATA")
    ccxt_timeframe: str = Field(default="1m", alias="CCXT_TIMEFRAME")
    ccxt_timeout: int = Field(default=10000, alias="CCXT_TIMEOUT_MS")

    database_url: str = Field(
        default="sqlite:///" + "backend/data/trades.db",
        alias="QUANTUM_TRADER_DATABASE_URL",
    )

    # Exchange credentials (optional)
    binance_api_key: str | None = Field(default=None, alias="BINANCE_API_KEY")
    binance_api_secret: str | None = Field(default=None, alias="BINANCE_API_SECRET")

    coinbase_api_key: str | None = Field(default=None, alias="COINBASE_API_KEY")
    coinbase_api_secret: str | None = Field(default=None, alias="COINBASE_API_SECRET")

    kucoin_api_key: str | None = Field(default=None, alias="KUCOIN_API_KEY")
    kucoin_api_secret: str | None = Field(default=None, alias="KUCOIN_API_SECRET")

    cryptopanic_key: str | None = Field(default=None, alias="CRYPTOPANIC_KEY")

    # Twitter / X credentials
    x_bearer_token: str | None = Field(default=None, alias="X_BEARER_TOKEN")
    x_api_key: str | None = Field(default=None, alias="X_API_KEY")
    x_api_secret: str | None = Field(default=None, alias="X_API_SECRET")
    x_access_token: str | None = Field(default=None, alias="X_ACCESS_TOKEN")
    x_access_secret: str | None = Field(default=None, alias="X_ACCESS_SECRET")

    model_config = SettingsConfigDict(
        env_file=(".env", "backend/.env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def default_symbols(self) -> list[str]:
        raw = self.default_symbols_raw
        if isinstance(raw, str):
            return [token.strip() for token in raw.split(',') if token.strip()]
        if isinstance(raw, (list, tuple)):
            return [str(token).strip() for token in raw if str(token).strip()]
        return [f"BTC{self.default_quote}", f"ETH{self.default_quote}"]

    @property
    def mainbase_symbols(self) -> list[str]:
        raw = self.mainbase_symbols_raw
        if isinstance(raw, str):
            return [token.strip() for token in raw.split(',') if token.strip()]
        if isinstance(raw, (list, tuple)):
            return [str(token).strip() for token in raw if str(token).strip()]
        return []

    @property
    def layer1_symbols(self) -> list[str]:
        raw = self.layer1_symbols_raw
        if isinstance(raw, str):
            return [token.strip() for token in raw.split(',') if token.strip()]
        if isinstance(raw, (list, tuple)):
            return [str(token).strip() for token in raw if str(token).strip()]
        return []

    @property
    def layer2_symbols(self) -> list[str]:
        raw = self.layer2_symbols_raw
        if isinstance(raw, str):
            return [token.strip() for token in raw.split(',') if token.strip()]
        if isinstance(raw, (list, tuple)):
            return [str(token).strip() for token in raw if str(token).strip()]
        return []

    @property
    def resolved_futures_quote(self) -> str:
        return self.futures_quote or self.default_quote


@lru_cache(maxsize=1)
def load_config() -> Settings:
    """Return a cached settings object."""

    return Settings()


settings: Settings = load_config()


def make_pair(base: str, quote: str | None = None) -> str:
    q = quote or settings.resolved_futures_quote
    return f"{base}{q}"


def masked_config_summary(cfg: Settings) -> Dict[str, Any]:
    return {
        "has_binance_keys": bool(cfg.binance_api_key and cfg.binance_api_secret),
        "has_coinbase_keys": bool(cfg.coinbase_api_key and cfg.coinbase_api_secret),
        "has_kucoin_keys": bool(cfg.kucoin_api_key and cfg.kucoin_api_secret),
        "live_market_data": bool(cfg.enable_live_market_data),
    }


DEFAULT_EXCHANGE = settings.default_exchange
FUTURES_QUOTE = settings.resolved_futures_quote
DEFAULT_SYMBOLS = settings.default_symbols
MAINBASE_SYMBOLS = settings.mainbase_symbols
LAYER1_SYMBOLS = settings.layer1_symbols
LAYER2_SYMBOLS = settings.layer2_symbols


__all__ = [
    "DEFAULT_EXCHANGE",
    "FUTURES_QUOTE",
    "DEFAULT_SYMBOLS",
    "Settings",
    "load_config",
    "make_pair",
    "masked_config_summary",
    "settings",
]
