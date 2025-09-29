
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

    model_config = SettingsConfigDict(
        env_file=(".env", "backend/.env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

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
DEFAULT_QUOTE = settings.default_quote
FUTURES_QUOTE = settings.resolved_futures_quote


__all__ = [
    "DEFAULT_EXCHANGE",
    "DEFAULT_QUOTE",
    "FUTURES_QUOTE",
    "Settings",
    "load_config",
    "make_pair",
    "masked_config_summary",
    "settings",
]
