from typing import Optional, Dict


class Config:
    binance_api_key: Optional[str]
    binance_api_secret: Optional[str]
    coinbase_api_key: Optional[str]
    coinbase_api_secret: Optional[str]
    kucoin_api_key: Optional[str]
    kucoin_api_secret: Optional[str]
    cryptopanic_key: Optional[str]
    x_api_key: Optional[str]
    x_api_secret: Optional[str]
    x_bearer_token: Optional[str]


def load_config() -> Config: ...


def masked_config_summary(cfg: Config) -> Dict[str, Dict[str, Optional[str]]]: ...
