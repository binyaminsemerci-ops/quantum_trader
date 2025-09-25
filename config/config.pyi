from typing import Dict, Any

DEFAULT_EXCHANGE: str

def load_config() -> Dict[str, Any]: ...

settings: Dict[str, Any]

__all__ = ["DEFAULT_EXCHANGE", "load_config", "settings"]
