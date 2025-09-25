from typing import Any

DEFAULT_EXCHANGE: str

def load_config() -> Any: ...

# `settings` is intentionally left as Any so external modules may access
# attributes or dict keys without mypy raising attr-defined errors in CI.
settings: Any

__all__ = ["DEFAULT_EXCHANGE", "load_config", "settings"]
