from typing import Any

DEFAULT_EXCHANGE: str
FUTURES_QUOTE: Any
DEFAULT_SYMBOLS: Any

def make_pair(base: str, quote: Any | None = ...) -> Any: ...
def load_config() -> Any: ...

# `settings` is intentionally left as Any so external modules may access
# attributes or dict keys without mypy raising attr-defined errors in CI.
settings: Any

def masked_config_summary(cfg: Any) -> Any: ...

__all__ = [
    "DEFAULT_EXCHANGE",
    "FUTURES_QUOTE",
    "DEFAULT_SYMBOLS",
    "make_pair",
    "load_config",
    "settings",
    "masked_config_summary",
]
